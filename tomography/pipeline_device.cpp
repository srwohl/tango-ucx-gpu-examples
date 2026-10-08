// SPDX-License-Identifier: LGPL-3.0-or-later
// Ordinary Tango device servers; all data links use the installed public tango-ucx interface.
#include <tango-ucx/tango.h>
#include <tango/tango.h>
#include <cuda_runtime_api.h>
#include <pybind11/embed.h>
#include <nlohmann/json.hpp>
#include "reconstruction_settings.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
namespace py = pybind11;
using namespace TangoUcx;
using Json = nlohmann::json;
using namespace std::chrono_literals;
const std::string field_dtype =
    "[('kind', '<u8'), ('projection', '<u8'), ('theta', '<f8'), "
    "('scan_id', '<u8'), ('calibration_id', '<u8')]";
struct Fields {
    std::uint64_t kind, projection;
    double theta;
    std::uint64_t scan_id, calibration_id;
};
static_assert(sizeof(Fields) == 40);
const std::string block_field_dtype = field_dtype.substr(0, field_dtype.size() - 1) +
    ", ('settings_revision', '<u8'), ('slice_start', '<u8'), ('slice_count', '<u8')]";
struct BlockFields {
    Fields frame;
    std::uint64_t settings_revision, slice_start, slice_count;
};
static_assert(sizeof(BlockFields) == 64);
struct Options {
    std::string role, upstream;
    std::filesystem::path file;
    Json scan;
    int gpu = 0, delay_ms = 0, iterations = 40, scan_period_ms = 0;
    int transport_batch = 1, processing_batch = 1;
    int reconstructors = 1; // above one, the reconstruction devices pull whole scans from correct
    int chains = 1; // above one, each decompress device pulls whole scans and feeds its own chain
    std::string processing_mode = "scalar";
    bool allow_gpu_over_tcp = false;
    std::uint64_t scan_count = 1; // zero: keep the publisher alive until Stop
    std::uint64_t budget = 256ull << 10;
} cfg;
void check(cudaError_t result) {
    if(result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
Json validated_reconstruction(const Json &options) {
    py::gil_scoped_acquire gil;
    try {
        auto json = py::module_::import("json");
        auto result = py::module_::import("reconstruction").attr("live_configuration")(
            json.attr("loads")(options.dump()), cfg.scan.at("columns").get<std::uint64_t>());
        // Reject live layouts exceeding launch-time host/pinned limits before queueing.
        py::module_::import("host_buffering").attr("memory_plan")(
            json.attr("loads")(cfg.scan.dump()), result);
        return Json::parse(json.attr("dumps")(result).cast<std::string>());
    } catch(const py::error_already_set &error) {
        throw std::runtime_error(error.what());
    }
}

class Device : public TANGO_BASE_CLASS {
public:
    Device(Tango::DeviceClass *owner, const std::string &name)
        : TANGO_BASE_CLASS(owner, name.c_str()) { init_device(); }
    ~Device() override { delete_device(); }
    void init_device() override {
        const auto buffers = cfg.scan.value("buffering", Json::object());
        block_output = cfg.role == "reconstruct" && buffers.value("output_mode", "volume") == "blocks";
        if(cfg.role == "reconstruct") {
            reconstruction = std::make_unique<ReconstructionSettings>(validated_reconstruction(
                cfg.scan.value("reconstruction", Json{{"algorithm", "sirt"}, {"iterations", cfg.iterations}})));
        }
        const auto rows = cfg.scan.at("rows").get<std::uint64_t>();
        const auto cols = cfg.scan.at("columns").get<std::uint64_t>();
        Description d;
        d.field_bytes = sizeof(Fields);
        d.field_dtype = field_dtype;
        if(block_output) {
            d.field_bytes = sizeof(BlockFields);
            d.field_dtype = block_field_dtype;
        }
        Json text = cfg.scan;
        text.erase("frames");
        text.erase("theta");
        text["role"] = cfg.role;
        text["scan_count"] = cfg.scan_count;
        d.application_text = text.dump();
        const bool detector_float = cfg.scan.value("element", "u16") == "f32";
        std::uint64_t bytes = rows * cols;
        if(cfg.role == "source") {
            d.element = Element::Bytes;
            d.max_payload_bytes = cfg.scan.at("max_compressed_bytes");
            bytes = d.max_payload_bytes;
        } else {
            d.element = cfg.role == "decompress" && !detector_float ? Element::U16 : Element::F32;
            d.shape = {rows, cols};
            bytes *= cfg.role == "decompress" && !detector_float ? 2 : 4;
            if(cfg.role == "reconstruct") {
                output_rows = block_output ? buffers.at("output_block_rows").get<std::uint64_t>() : rows;
                d.shape = {output_rows, cols, cols};
                bytes = output_rows * cols * cols * 4;
            }
        }
        PublisherLimits limits;
        // Independent archive and live-view subscriptions join their respective publishers.
        limits.max_sessions = cfg.role == "source" ? 1 + cfg.chains : cfg.role == "reconstruct" ? 2 :
                              cfg.role == "correct" ? cfg.reconstructors : 1;
        const auto output_margin = cfg.role == "decompress" ?
            std::max(8, 2 * cfg.processing_batch) : 8;
        limits.budget = std::max<std::uint64_t>(cfg.budget, bytes * output_margin + (192ull << 10));
        publisher_budget = limits.budget;
        output_payload_bytes = bytes;
        limits.allow_every = true;
        limits.pressure_timeout = 60s;
        limits.gpu = cfg.role == "source" ? -1 : cfg.gpu;
        publisher = std::make_unique<Publisher>(std::move(d), limits);
        attach_publisher(*this, *publisher);
        set_state(Tango::ON);
        set_status("Configured " + cfg.role + "; Arm links upstream, Start begins processing");
    }
    void delete_device() override {
        worker.request_stop();
        if(input) input->close(std::chrono::steady_clock::now() + 2s);
        worker = {};
        input.reset();
        proxy.reset();
        detach_publisher(*this);
        publisher.reset();
    }
    void arm() {
        std::lock_guard lock(commands);
        if(armed || started) throw std::runtime_error("Arm requires a fresh device");
        if(cfg.role != "source") {
            proxy = std::make_unique<Tango::DeviceProxy>(cfg.upstream);
            SubscriptionOptions receive;
            receive.gpu = cfg.gpu;
            receive.budget = cfg.budget;
            receive.batch = cfg.transport_batch;
            receive.allow_gpu_over_tcp = cfg.allow_gpu_over_tcp;
            receive.label = "tomography-" + cfg.role;
            // Correct publishes one frame per projection, so a range of them is one scan. The
            // receive ring has to hold it: a puller takes a range only with room for all of it.
            input = std::make_unique<Subscription>(pull_range() ?
                pull(*proxy, pull_range(), receive) : every(*proxy, receive));
            const auto &d = input->description();
            const auto meta = Json::parse(d.application_text);
            const auto expected = cfg.role == "decompress" ? "source" :
                                  cfg.role == "correct" ? "decompress" : "correct";
            if(d.field_bytes != sizeof(Fields) || d.field_dtype != field_dtype ||
               meta.at("role") != expected || meta.at("scan_id") != cfg.scan.at("scan_id") ||
               meta.at("rows") != cfg.scan.at("rows") ||
               meta.at("columns") != cfg.scan.at("columns") ||
               meta.at("angles") != cfg.scan.at("angles") ||
               meta.value("element", "u16") != cfg.scan.value("element", "u16"))
                throw std::runtime_error("upstream description does not match this stage");
            const auto expected_element = cfg.role == "decompress" ? Element::Bytes :
                cfg.role == "correct" && cfg.scan.value("element", "u16") == "u16" ?
                    Element::U16 : Element::F32;
            if(d.element != expected_element)
                throw std::runtime_error("upstream payload element does not match this stage");
        }
        armed = true;
        set_state(Tango::STANDBY);
    }
    void start() {
        std::lock_guard lock(commands);
        if(!armed || started) throw std::runtime_error("Start requires Arm and runs once");
        publisher->ucx_start();
        started = true;
        set_state(Tango::RUNNING);
        worker = std::jthread([this](std::stop_token stop) {
            try {
                if(cfg.role == "source") produce(stop);
                else process(stop);
                {
                    std::lock_guard guard(commands);
                    if(reconstruction) reconstruction->finish();
                }
                if(!stop.stop_requested()) {
                    publisher->finish();
                    set_state(Tango::ON);
                    set_status("Finished " + cfg.role);
                }
            } catch(const std::exception &error) {
                std::lock_guard guard(commands);
                failure = error.what();
                if(reconstruction) reconstruction->finish();
                set_state(Tango::FAULT);
                set_status(failure);
            }
        });
    }
    std::string report() {
        std::lock_guard guard(commands);
        const auto h = publisher->health();
        return Json{{"role", cfg.role}, {"processed", processed.load()},
                    {"transport_batch", cfg.transport_batch}, {"processing_batch", cfg.processing_batch},
                    {"processing_mode", cfg.processing_mode},
                    {"completed_scans", completed_scans.load()},
                    {"published", published.load()}, {"pressure", h.pressure},
                    {"slot_wait_ns", slot_wait_ns.load()},
                    {"publisher_budget_bytes", publisher_budget},
                    {"payload_bytes", output_payload_bytes},
                    {"publisher_slots_free", h.slots_free}, {"publisher_slots_held", h.slots_held},
                    {"reconstruct_ns", reconstruct_ns.load()},
                    {"publisher_payload_storage_upper_bytes",
                        std::min<std::uint64_t>(1024, publisher_budget / output_payload_bytes) * output_payload_bytes},
                    {"failure", failure.empty() ? h.failure : failure},
                    {"quarantined_bytes", h.quarantined_bytes},
                    {"input_transport", input ? input->health().transport : ""},
                    {"input_failure", input ? input->health().failure : ""}}.dump();
    }
    std::string reconstruction_settings() {
        std::lock_guard guard(commands);
        require_reconstruction();
        return settings_state().dump();
    }
    std::string configure_reconstruction(const std::string &text) {
        require_reconstruction();
        // Acquire the GIL before the command mutex: the worker snapshots with its GIL held.
        const auto options = validated_reconstruction(Json::parse(text));
        std::lock_guard guard(commands);
        reconstruction->configure(options);
        return settings_state().dump();
    }
    std::string reconstruction_for_scan(const std::string &text) {
        const auto scan_id = Json::parse(text).get<std::uint64_t>();
        std::lock_guard guard(commands);
        require_reconstruction();
        return reconstruction->for_scan(scan_id).dump();
    }
    void stop_acquisition() {
        std::lock_guard guard(commands);
        if(cfg.role != "source" || !started)
            throw std::runtime_error("Stop requires a running source");
        finish_requested = true; // Complete this scan, then send End through every stage.
    }
private:
    void require_reconstruction() const {
        if(cfg.role != "reconstruct")
            throw std::runtime_error("settings are available only on the reconstruction device");
    }
    Json settings_state() const {
        auto state = reconstruction->state(cfg.scan.at("columns").get<std::uint64_t>());
        state["output_mode"] = block_output ? "blocks" : "volume";
        state["output_capacity_slices"] = output_rows;
        return state;
    }
    std::uint32_t pull_range() const {
        // The source publishes a dark, a flat and the projections of each scan.
        if(cfg.role == "decompress" && cfg.chains > 1)
            return cfg.scan.at("angles").get<std::uint32_t>() + 2;
        return cfg.role == "reconstruct" && cfg.reconstructors > 1 ?
            cfg.scan.at("angles").get<std::uint32_t>() : 0;
    }
    Slot acquire(std::stop_token stop) {
        const auto begin = std::chrono::steady_clock::now();
        while(!stop.stop_requested()) {
            auto slot = publisher->acquire(std::chrono::steady_clock::now() + 100ms);
            if(slot) {
                slot_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count());
                return std::move(*slot);
            }
        }
        throw std::runtime_error("processing cancelled");
    }
    void produce(std::stop_token stop) {
        std::ifstream data(cfg.file.parent_path() / "compressed.bin", std::ios::binary);
        if(!data) throw std::runtime_error("cannot open compressed.bin");
        for(std::uint64_t cycle = 0; !cfg.scan_count || cycle < cfg.scan_count; ++cycle) {
          if(cycle && finish_requested.load()) break;
          const auto next_scan = std::chrono::steady_clock::now() +
                                 std::chrono::milliseconds(cfg.scan_period_ms);
          data.clear();
          data.seekg(0);
          for(const auto &entry : cfg.scan.at("frames")) {
            auto slot = acquire(stop);
            const auto bytes = entry.at("bytes").get<std::size_t>();
            if(bytes > slot.payload().size()) throw std::runtime_error("compressed frame too large");
            data.read(reinterpret_cast<char *>(slot.payload().data()),
                      static_cast<std::streamsize>(bytes));
            if(!data) throw std::runtime_error("truncated compressed scan");
            Fields fields{entry.at("kind"), entry.at("projection"), entry.at("theta"),
                          cfg.scan.at("scan_id").get<std::uint64_t>() + cycle,
                          cfg.scan.at("calibration_id").get<std::uint64_t>() + cycle};
            std::memcpy(slot.fields().data(), &fields, sizeof(fields));
            std::move(slot).publish_bytes(bytes);
            ++processed;
            ++published;
          }
          ++completed_scans;
          if(cfg.scan_count && cycle + 1 == cfg.scan_count) break;
          while(!stop.stop_requested() && !finish_requested.load() &&
                std::chrono::steady_clock::now() < next_scan)
              std::this_thread::sleep_for(10ms);
          if(stop.stop_requested() || finish_requested.load()) break;
        }
    }
    void process(std::stop_token stop) {
        check(cudaSetDevice(cfg.gpu));
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            py::gil_scoped_acquire gil;
            try {
                auto meta = py::module_::import("json").attr("loads")(cfg.scan.dump());
                auto processor = py::module_::import("processors").attr("Processor")(
                    cfg.role, meta, cfg.gpu, reinterpret_cast<std::uintptr_t>(stream), cfg.iterations);
                std::uint64_t current_scan = 0, calibration = 0, settings_revision = 0, block_rows = 0;
                std::uint64_t next_index = 0;
                const std::uint64_t range = pull_range();
                try {
                    while(!stop.stop_requested()) {
                        auto batch = [&] {
                            py::gil_scoped_release release;
                            return input->read(std::chrono::steady_clock::now() + 100ms);
                        }();
                        if(!batch) {
                            if(input->outcome() == Outcome::End) break;
                            if(input->outcome()) throw std::runtime_error("upstream ended without End");
                            continue;
                        }
                        struct Frame { Fields fields{}; std::uint64_t timestamp = 0, index = 0; };
                        std::vector<Frame> frames(batch->frames());
                        // Public records have two uint64s followed by our 40-byte fields.
                        const auto records = static_cast<const std::byte *>(batch->records());
                        for(std::size_t i = 0; i < frames.size(); ++i) {
                            // A puller's next range is any later one; frames within it are consecutive.
                            frames[i].index = batch->index(i);
                            if(range && next_index % range == 0 ?
                                   frames[i].index % range || frames[i].index < next_index :
                                   frames[i].index != next_index)
                                throw std::runtime_error("missing or out-of-order upstream frame");
                            next_index = frames[i].index + 1;
                            const auto record = records + i * (16 + sizeof(Fields));
                            std::memcpy(&frames[i].timestamp, record + 8, 8);
                            std::memcpy(&frames[i].fields, record + 16, sizeof(Fields));
                        }
                        auto begin_frame = [&](const Frame &first) {
                            const auto &fields = first.fields;
                            if(fields.scan_id != current_scan) {
                                if(current_scan) processor.attr("finish")();
                                // Within a chain the frames are consecutive but its scans are not.
                                const auto base = cfg.scan.at("scan_id").get<std::uint64_t>();
                                if(cfg.chains > 1 && (fields.scan_id < base || fields.scan_id <= current_scan))
                                    throw std::runtime_error("scan identity did not advance");
                                const auto scan_number = range ? first.index / range :
                                    cfg.chains > 1 ? fields.scan_id - base : completed_scans.load();
                                if(fields.scan_id != cfg.scan.at("scan_id").get<std::uint64_t>() +
                                       scan_number || fields.projection != 0 ||
                                   fields.kind != (cfg.role == "reconstruct" ? 2 : 0) ||
                                   fields.calibration_id != cfg.scan.at("calibration_id").get<std::uint64_t>() +
                                       scan_number)
                                    throw std::runtime_error("missing scan boundary or calibration");
                                processor.attr("begin_scan")();
                                if(cfg.role == "reconstruct") {
                                    Json options;
                                    {
                                        std::lock_guard guard(commands);
                                        options = reconstruction->begin_scan(fields.scan_id);
                                        settings_revision = reconstruction->for_scan(fields.scan_id).at("revision");
                                    }
                                    block_rows = std::min(options.value("slices_per_block", std::uint64_t(0)),
                                                          cfg.scan.at("rows").get<std::uint64_t>());
                                    processor.attr("configure_reconstruction")(
                                        py::module_::import("json").attr("loads")(options.dump()));
                                }
                                current_scan = fields.scan_id;
                                calibration = fields.calibration_id;
                            }
                            if(fields.calibration_id != calibration)
                                throw std::runtime_error("calibration changed within a scan");
                        };
                        auto complete_frame = [&](const Fields &fields) {
                            ++processed;
                            if(fields.kind == 2 && fields.projection + 1 ==
                               cfg.scan.at("angles").get<std::uint64_t>()) {
                                processor.attr("finish")();
                                ++completed_scans;
                                if(cfg.role == "reconstruct")
                                    reconstruct_ns = processor.attr("reconstruct_ns").cast<std::uint64_t>();
                            }
                        };
                        auto view = std::move(*batch).gpu_view(GpuStream{stream});
                        for(std::size_t frame = 0; frame < frames.size();) {
                            const auto &fields = frames[frame].fields;
                            const auto timestamp = frames[frame].timestamp;
                            begin_frame(frames[frame]);
                            if(cfg.role == "decompress" && cfg.processing_mode == "batched") {
                                auto end = std::min(frames.size(), frame + cfg.processing_batch);
                                for(auto i = frame + 1; i < end; ++i)
                                    if(frames[i].fields.scan_id != fields.scan_id) { end = i; break; }
                                if(end - frame > 1) {
                                    std::vector<Slot> outputs;
                                    outputs.reserve(end - frame);
                                    py::list arguments;
                                    for(auto i = frame; i < end; ++i) {
                                        if(frames[i].fields.calibration_id != calibration)
                                            throw std::runtime_error("calibration changed within a scan");
                                        {
                                            py::gil_scoped_release release;
                                            outputs.emplace_back(acquire(stop));
                                        }
                                        auto &slot = outputs.back();
                                        auto destination = slot.gpu_payload(GpuStream{stream});
                                        std::memcpy(slot.fields().data(), &frames[i].fields, sizeof(Fields));
                                        arguments.append(py::make_tuple(
                                            reinterpret_cast<std::uintptr_t>(view.payload(i)), view.payload_bytes(i),
                                            reinterpret_cast<std::uintptr_t>(destination), frames[i].fields.kind,
                                            frames[i].fields.projection, frames[i].fields.theta));
                                    }
                                    processor.attr("consume_many")(arguments);
                                    if(end == frames.size()) view = GpuView{};
                                    for(auto i = frame; i < end; ++i) {
                                        std::move(outputs[i - frame]).publish(frames[i].timestamp);
                                        ++published;
                                        complete_frame(frames[i].fields);
                                    }
                                    frame = end;
                                    continue;
                                }
                            }
                            if(cfg.delay_ms) {
                                py::gil_scoped_release release;
                                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.delay_ms));
                            }
                            const bool emits = cfg.role == "decompress" ||
                                (cfg.role == "correct" && fields.kind == 2) ||
                                (cfg.role == "reconstruct" && !block_output && fields.projection + 1 ==
                                    cfg.scan.at("angles").get<std::uint64_t>());
                            std::optional<Slot> output;
                            void *destination = nullptr;
                            if(emits) {
                                {
                                    py::gil_scoped_release release;
                                    output.emplace(acquire(stop));
                                }
                                destination = output->gpu_payload(GpuStream{stream});
                                std::memcpy(output->fields().data(), &fields, sizeof(fields));
                            }
                            processor.attr("consume")(
                                reinterpret_cast<std::uintptr_t>(view.payload(frame)), view.payload_bytes(frame),
                                reinterpret_cast<std::uintptr_t>(destination),
                                fields.kind, fields.projection, fields.theta);
                            // One receive view owns the whole transport batch. Release after
                            // its last queued read, before final-frame output credit waits.
                            if(frame + 1 == frames.size()) view = GpuView{};
                            if(block_output && fields.projection + 1 == cfg.scan.at("angles").get<std::uint64_t>()) {
                                const auto rows = cfg.scan.at("rows").get<std::uint64_t>();
                                if(!block_rows || block_rows > output_rows)
                                    throw std::runtime_error("reconstruction block exceeds fixed output capacity");
                                for(std::uint64_t begin = 0; begin < rows; begin += block_rows) {
                                    const auto count = std::min(block_rows, rows - begin);
                                    auto slot = [&] {
                                        py::gil_scoped_release release;
                                        return acquire(stop);
                                    }();
                                    auto pointer = slot.gpu_payload(GpuStream{stream});
                                    processor.attr("reconstruct_block")(
                                        reinterpret_cast<std::uintptr_t>(pointer), begin, count);
                                    BlockFields block_fields{fields, settings_revision, begin, count};
                                    std::memcpy(slot.fields().data(), &block_fields, sizeof(block_fields));
                                    std::move(slot).publish(timestamp);
                                    ++published;
                                }
                            }
                            // Receive completion follows the last queued read. Publication separately
                            // follows the writes to this GPU source slot on the same CUDA stream.
                            if(output) {
                                std::move(*output).publish(timestamp);
                                ++published;
                            }
                            complete_frame(fields);
                            ++frame;
                        }
                    }
                    if(!stop.stop_requested()) processor.attr("finish")();
                    processor.attr("drain")();
                    check(cudaStreamSynchronize(stream));
                } catch(...) {
                    // Python-owned scratch/maps stay alive until queued GPU work completes.
                    // A prefetched block also owns work on a separate H2D stream.
                    try { processor.attr("drain")(); } catch(...) {}
                    (void)cudaStreamSynchronize(stream);
                    throw;
                }
            } catch(const py::error_already_set &error) {
                throw std::runtime_error(error.what());
            }
        } catch(...) {
            (void)cudaStreamDestroy(stream);
            throw;
        }
        check(cudaStreamDestroy(stream));
    }
    std::mutex commands;
    std::unique_ptr<Publisher> publisher;
    std::unique_ptr<Tango::DeviceProxy> proxy;
    std::unique_ptr<Subscription> input;
    std::jthread worker;
    bool armed = false, started = false;
    bool block_output = false;
    std::uint64_t output_rows = 0, publisher_budget = 0, output_payload_bytes = 0;
    std::unique_ptr<ReconstructionSettings> reconstruction;
    std::atomic<std::uint64_t> processed{0}, published{0}, slot_wait_ns{0}, completed_scans{0};
    std::atomic<std::uint64_t> reconstruct_ns{0};
    std::atomic<bool> finish_requested{false};
    std::string failure;
};
class Command : public Tango::Command {
public:
    explicit Command(const char *name, Tango::CmdArgType out = Tango::DEV_VOID,
                     Tango::CmdArgType in = Tango::DEV_VOID)
        : Tango::Command(name, in, out) {}
    CORBA::Any *execute(Tango::DeviceImpl *impl, const CORBA::Any &argument) override {
        auto &device = *static_cast<Device *>(impl);
        try {
            if(get_name() == "Arm") device.arm();
            else if(get_name() == "Start") device.start();
            else if(get_name() == "Stop") device.stop_acquisition();
            else if(get_name() == "GetReconstruction")
                return insert(CORBA::string_dup(device.reconstruction_settings().c_str()));
            else if(get_name() == "ConfigureReconstruction" || get_name() == "ReconstructionForScan") {
                Tango::DevString text;
                extract(argument, text);
                const auto result = get_name() == "ConfigureReconstruction" ?
                    device.configure_reconstruction(text) : device.reconstruction_for_scan(text);
                return insert(CORBA::string_dup(result.c_str()));
            }
            else return insert(CORBA::string_dup(device.report().c_str()));
        } catch(const std::exception &error) {
            Tango::Except::throw_exception("PipelineRefused", error.what(), "pipeline_device");
        }
        return insert();
    }
};
class DeviceClass : public Tango::DeviceClass {
public:
    DeviceClass() : Tango::DeviceClass(std::string("GpuPipeline")) {}
    void command_factory() override {
        command_list.push_back(new Command("Arm"));
        command_list.push_back(new Command("Start"));
        command_list.push_back(new Command("Stop"));
        command_list.push_back(new Command("Report", Tango::DEV_STRING));
        command_list.push_back(new Command("GetReconstruction", Tango::DEV_STRING));
        command_list.push_back(new Command("ConfigureReconstruction", Tango::DEV_STRING, Tango::DEV_STRING));
        command_list.push_back(new Command("ReconstructionForScan", Tango::DEV_STRING, Tango::DEV_STRING));
        install_ucx_commands(command_list);
    }
    void attribute_factory(std::vector<Tango::Attr *> &list) override { install_ucx_attributes(list); }
    void device_factory(const Tango::DevVarStringArray *names) override {
        for(CORBA::ULong i = 0; i < names->length(); ++i)
            device_list.push_back(new Device(this, std::string((*names)[i])));
        for(auto *device : device_list) export_device(device, device->get_name().c_str());
    }
};
class Server : public Tango::DServer {
public:
    using Tango::DServer::DServer;
    void class_factory() override { add_class(new DeviceClass); }
};
} // namespace

int main(int argc, char **argv) {
    try {
        std::vector<char *> args{argv[0]};
        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if(arg.starts_with("--")) {
                if(++i == argc) throw std::runtime_error("missing value for " + arg);
                if(arg == "--role") cfg.role = argv[i];
                else if(arg == "--scan") cfg.file = argv[i];
                else if(arg == "--upstream") cfg.upstream = argv[i];
                else if(arg == "--gpu") cfg.gpu = std::stoi(argv[i]);
                else if(arg == "--allow-gpu-over-tcp") {
                    const std::string value = argv[i];
                    if(value != "0" && value != "1")
                        throw std::runtime_error("--allow-gpu-over-tcp needs 0 or 1");
                    cfg.allow_gpu_over_tcp = value == "1";
                }
                else if(arg == "--budget") cfg.budget = std::stoull(argv[i]);
                else if(arg == "--transport-batch") cfg.transport_batch = std::stoi(argv[i]);
                else if(arg == "--processing-batch") cfg.processing_batch = std::stoi(argv[i]);
                else if(arg == "--reconstructors") cfg.reconstructors = std::stoi(argv[i]);
                else if(arg == "--chains") cfg.chains = std::stoi(argv[i]);
                else if(arg == "--processing-mode") cfg.processing_mode = argv[i];
                else if(arg == "--delay-ms") cfg.delay_ms = std::stoi(argv[i]);
                else if(arg == "--iterations") cfg.iterations = std::stoi(argv[i]);
                else if(arg == "--scans") cfg.scan_count = std::stoull(argv[i]);
                else if(arg == "--scan-period-ms") cfg.scan_period_ms = std::stoi(argv[i]);
                else throw std::runtime_error("unknown option " + arg);
            } else args.push_back(argv[i]);
        }
        if(cfg.role != "source" && cfg.role != "decompress" &&
           cfg.role != "correct" && cfg.role != "reconstruct")
            throw std::runtime_error("--role needs source, decompress, correct or reconstruct");
        // MAX_BATCH in pipeline_control.py: twice a batch of output slots within the publisher's 1024.
        if(cfg.transport_batch < 1 || cfg.transport_batch > 512 ||
           cfg.processing_batch < 1 || cfg.processing_batch > 512)
            throw std::runtime_error("transport and processing batch sizes must be 1 to 512");
        if(cfg.chains < 1 || cfg.chains > 63)
            throw std::runtime_error("--chains needs 1 to 63");
        if(cfg.reconstructors < 1 || cfg.reconstructors > 64)
            throw std::runtime_error("--reconstructors needs 1 to 64 publisher sessions");
        if(cfg.processing_mode != "scalar" && cfg.processing_mode != "batched")
            throw std::runtime_error("--processing-mode needs scalar or batched");
        std::ifstream scan(cfg.file);
        scan >> cfg.scan;
        const auto element = cfg.scan.value("element", "u16");
        if(element != "u16" && element != "f32")
            throw std::runtime_error("scan detector element needs u16 or f32");
        py::scoped_interpreter interpreter;
        py::module_::import("sys").attr("path").attr("insert")(0, TOMOGRAPHY_MODULE_DIR);
        py::gil_scoped_release release;
        auto *util = Tango::Util::init(static_cast<int>(args.size()), args.data());
        util->register_dserver_constructor(
            [](Tango::DeviceClass *c, const std::string &name, const std::string &description,
               Tango::DevState state, const std::string &status) -> Tango::DServer *
            { return new Server(c, name.c_str(), description.c_str(), state, status.c_str()); });
        util->server_init();
        util->server_run();
        util->server_cleanup();
        return 0;
    } catch(const Tango::DevFailed &error) { Tango::Except::print_exception(error); }
      catch(const std::exception &error) { std::cerr << error.what() << std::endl; }
    return 1;
}
