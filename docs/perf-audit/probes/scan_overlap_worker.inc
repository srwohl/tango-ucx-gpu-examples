    void process_overlap(std::stop_token stop) {
        if(cfg.reconstructors != 1 || cfg.chains != 1 || block_output)
            throw std::runtime_error("scan overlap requires one reconstruction worker and one chain");
        const auto buffers_text = std::getenv("TANGO_SCAN_BUFFERS");
        const auto buffer_count = buffers_text ? std::stoi(buffers_text) : 2;
        if(buffer_count != 1 && buffer_count != 2)
            throw std::runtime_error("scan overlap supports one or two buffers");
        struct Job {
            int buffer;
            Fields fields;
            std::uint64_t timestamp;
        };
        check(cudaSetDevice(cfg.gpu));
        cudaStream_t assembly_stream = nullptr, reconstruction_stream = nullptr;
        std::vector<cudaEvent_t> ready_events(buffer_count, nullptr);
        py::object assembler;
        std::mutex queue_mutex;
        std::condition_variable changed;
        std::deque<int> free_buffers;
        std::deque<Job> ready;
        std::exception_ptr reconstruction_error;
        bool input_done = false;
        std::atomic<bool> reconstructing{false};
        std::uint64_t groups = 0, concurrent_groups = 0, partial_groups = 0, wrap_splits = 0;
        std::uint64_t assembled = 0, assembly_ns = 0, buffer_wait_ns = 0;
        std::size_t peak_ready = 0, peak_busy = 0;
        std::jthread reconstructor;
        auto cleanup = [&] {
            if(assembly_stream) (void)cudaStreamSynchronize(assembly_stream);
            if(reconstruction_stream) (void)cudaStreamSynchronize(reconstruction_stream);
            {
                py::gil_scoped_acquire gil;
                assembler = py::object{};
            }
            for(auto event : ready_events) if(event) (void)cudaEventDestroy(event);
            if(assembly_stream) (void)cudaStreamDestroy(assembly_stream);
            if(reconstruction_stream) (void)cudaStreamDestroy(reconstruction_stream);
        };
        try {
            check(cudaStreamCreateWithFlags(&assembly_stream, cudaStreamNonBlocking));
            check(cudaStreamCreateWithFlags(&reconstruction_stream, cudaStreamNonBlocking));
            for(int buffer = 0; buffer < buffer_count; ++buffer) {
                check(cudaEventCreateWithFlags(&ready_events[buffer], cudaEventDisableTiming));
                free_buffers.push_back(buffer);
            }
            {
                py::gil_scoped_acquire gil;
                auto meta = py::module_::import("json").attr("loads")(cfg.scan.dump());
                assembler = py::module_::import("scan_overlap").attr("ScanAssembly")(
                    meta, cfg.gpu, reinterpret_cast<std::uintptr_t>(assembly_stream),
                    reinterpret_cast<std::uintptr_t>(reconstruction_stream), cfg.iterations, buffer_count);
            }
            reconstructor = std::jthread([&](std::stop_token worker_stop) {
                try {
                    check(cudaSetDevice(cfg.gpu));
                    while(!worker_stop.stop_requested() && !stop.stop_requested()) {
                        std::unique_lock lock(queue_mutex);
                        changed.wait_for(lock, 100ms, [&] {
                            return !ready.empty() || input_done || worker_stop.stop_requested();
                        });
                        if(ready.empty()) {
                            if(input_done) break;
                            continue;
                        }
                        const auto job = ready.front();
                        ready.pop_front();
                        lock.unlock();
                        auto output = acquire(worker_stop);
                        auto destination = output.gpu_payload(GpuStream{reconstruction_stream});
                        std::memcpy(output.fields().data(), &job.fields, sizeof(Fields));
                        check(cudaStreamWaitEvent(reconstruction_stream, ready_events[job.buffer], 0));
                        reconstructing = true;
                        std::uint64_t elapsed;
                        {
                            py::gil_scoped_acquire gil;
                            try {
                                elapsed = assembler.attr("reconstruct")(
                                    job.buffer, reinterpret_cast<std::uintptr_t>(destination)).cast<std::uint64_t>();
                            } catch(const py::error_already_set &error) {
                                throw std::runtime_error(error.what());
                            }
                        }
                        check(cudaStreamSynchronize(reconstruction_stream));
                        reconstructing = false;
                        reconstruct_ns.fetch_add(elapsed);
                        std::move(output).publish(job.timestamp);
                        ++published;
                        ++completed_scans;
                        lock.lock();
                        free_buffers.push_back(job.buffer);
                        lock.unlock();
                        changed.notify_all();
                    }
                } catch(...) {
                    reconstructing = false;
                    std::lock_guard guard(queue_mutex);
                    reconstruction_error = std::current_exception();
                    changed.notify_all();
                }
            });
            const auto angles = cfg.scan.at("angles").get<std::uint64_t>();
            const auto frame_bytes = cfg.scan.at("rows").get<std::uint64_t>() *
                                     cfg.scan.at("columns").get<std::uint64_t>() * 4;
            std::uint64_t next_index = 0, projection = 0;
            std::optional<int> current_buffer;
            while(!stop.stop_requested()) {
                {
                    std::lock_guard guard(queue_mutex);
                    if(reconstruction_error) std::rethrow_exception(reconstruction_error);
                }
                auto batch = input->read(std::chrono::steady_clock::now() + 100ms);
                if(!batch) {
                    if(input->outcome() == Outcome::End) break;
                    if(input->outcome()) throw std::runtime_error("upstream ended without End");
                    continue;
                }
                struct Frame { Fields fields{}; std::uint64_t timestamp = 0; };
                std::vector<Frame> frames(batch->frames());
                const auto records = static_cast<const std::byte *>(batch->records());
                for(std::size_t index = 0; index < frames.size(); ++index) {
                    if(batch->index(index) != next_index++)
                        throw std::runtime_error("missing or out-of-order upstream frame");
                    const auto record = records + index * (16 + sizeof(Fields));
                    std::memcpy(&frames[index].timestamp, record + 8, 8);
                    std::memcpy(&frames[index].fields, record + 16, sizeof(Fields));
                }
                auto view = std::move(*batch).gpu_view(GpuStream{assembly_stream});
                for(std::size_t first = 0; first < frames.size();) {
                    if(!current_buffer) {
                        const auto begin = std::chrono::steady_clock::now();
                        std::unique_lock lock(queue_mutex);
                        while(free_buffers.empty() && !reconstruction_error && !stop.stop_requested())
                            changed.wait_for(lock, 100ms);
                        buffer_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin).count();
                        if(reconstruction_error) std::rethrow_exception(reconstruction_error);
                        if(stop.stop_requested()) throw std::runtime_error("processing cancelled");
                        current_buffer = free_buffers.front();
                        free_buffers.pop_front();
                        peak_busy = std::max(peak_busy, buffer_count - free_buffers.size());
                        lock.unlock();
                        Json options;
                        {
                            std::lock_guard guard(commands);
                            options = reconstruction->begin_scan(frames[first].fields.scan_id);
                        }
                        py::gil_scoped_acquire gil;
                        assembler.attr("begin_scan")(*current_buffer, frames[first].fields.scan_id,
                            py::module_::import("json").attr("loads")(options.dump()));
                    }
                    const auto scan_id = cfg.scan.at("scan_id").get<std::uint64_t>() + assembled;
                    const auto calibration = cfg.scan.at("calibration_id").get<std::uint64_t>() + assembled;
                    auto end = std::min<std::size_t>({frames.size(), first + cfg.processing_batch,
                                                    first + angles - projection});
                    for(auto index = first; index < end; ++index) {
                        const auto &fields = frames[index].fields;
                        if(fields.kind != 2 || fields.scan_id != scan_id || fields.calibration_id != calibration ||
                           fields.projection != projection + index - first || view.payload_bytes(index) != frame_bytes)
                            throw std::runtime_error("scan identity, calibration, projection or length changed");
                        if(reinterpret_cast<std::uintptr_t>(view.payload(index)) !=
                           reinterpret_cast<std::uintptr_t>(view.payload(first)) + (index - first) * frame_bytes) {
                            end = index;
                            ++wrap_splits;
                            break;
                        }
                    }
                    const auto begin = std::chrono::steady_clock::now();
                    const bool concurrent = reconstructing.load();
                    {
                        py::gil_scoped_acquire gil;
                        py::list arguments;
                        for(auto index = first; index < end; ++index)
                            arguments.append(py::make_tuple(reinterpret_cast<std::uintptr_t>(view.payload(index)),
                                view.payload_bytes(index), frames[index].fields.kind, frames[index].fields.projection,
                                frames[index].fields.theta));
                        assembler.attr("consume_group")(*current_buffer, arguments);
                    }
                    assembly_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - begin).count();
                    ++groups;
                    if(concurrent && reconstructing) ++concurrent_groups;
                    if(end - first < static_cast<std::size_t>(cfg.processing_batch)) ++partial_groups;
                    processed.fetch_add(end - first);
                    projection += end - first;
                    if(projection == angles) {
                        check(cudaEventRecord(ready_events[*current_buffer], assembly_stream));
                        std::lock_guard guard(queue_mutex);
                        ready.push_back(Job{*current_buffer, frames[end - 1].fields, frames[end - 1].timestamp});
                        peak_ready = std::max(peak_ready, ready.size());
                        ++assembled;
                        projection = 0;
                        current_buffer.reset();
                        changed.notify_all();
                    }
                    first = end;
                }
            }
            if(current_buffer && !stop.stop_requested())
                throw std::runtime_error("upstream ended with an incomplete scan");
            {
                std::lock_guard guard(queue_mutex);
                input_done = true;
            }
            changed.notify_all();
            if(stop.stop_requested()) reconstructor.request_stop();
            reconstructor.join();
            if(reconstruction_error) std::rethrow_exception(reconstruction_error);
            std::cerr << Json{{"scan_overlap_summary", true}, {"buffers", buffer_count},
                {"groups", groups}, {"groups_during_reconstruction", concurrent_groups},
                {"partial_groups", partial_groups}, {"ring_wrap_splits", wrap_splits},
                {"assembled_scans", assembled}, {"completed_scans", completed_scans.load()},
                {"peak_ready_scans", peak_ready}, {"peak_busy_buffers", peak_busy},
                {"assembly_host_ns", assembly_ns}, {"buffer_wait_ns", buffer_wait_ns},
                {"owned_sinogram_bytes", buffer_count * angles * frame_bytes}}.dump() << '\n';
        } catch(...) {
            reconstructor.request_stop();
            changed.notify_all();
            if(reconstructor.joinable()) reconstructor.join();
            cleanup();
            throw;
        }
        cleanup();
    }
