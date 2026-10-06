// SPDX-License-Identifier: LGPL-3.0-or-later
#include <tango-ucx/tango.h>
#include <tango/tango.h>
#include <nlohmann/json.hpp>
#include "calibration_pattern.h"
#ifdef EXAMPLES_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>

namespace
{
using namespace TangoUcx;
using Json = nlohmann::json;
using namespace std::chrono_literals;

class Device : public TANGO_BASE_CLASS
{
  public:
    Device(Tango::DeviceClass *c, const std::string &name) : TANGO_BASE_CLASS(c, name.c_str())
    {
        init_device();
    }
    ~Device() override { delete_device(); }
    void init_device() override
    {
        configure(R"({"source":"host"})");
        set_state(Tango::ON);
    }
    void delete_device() override
    {
        frames = {}; // requests stop and joins before detaching or freeing storage
        detach_publisher(*this);
        publisher.reset();
    }
    void configure(const std::string &document)
    {
        std::lock_guard lock(commands);
        if(running)
            throw std::runtime_error("wait for this run to finish before Configure");
        auto cfg = Json::parse(document);
        auto source = cfg.value("source", std::string("host"));
        int gpu = -1;
        if(source != "host")
        {
            if(!source.starts_with("cuda:") || source.size() == 5 ||
               source.find_first_not_of("0123456789", 5) != source.npos)
                throw std::invalid_argument("source must be host or cuda:N");
            gpu = std::stoi(source.substr(5));
#ifndef EXAMPLES_CUDA
            throw std::invalid_argument("this publisher was built without CUDA");
#endif
        }
        auto sizes = cfg.value("lengths", std::vector<std::uint64_t>{1, 129, 513, 65539, 262144});
        if(sizes.empty() || std::ranges::any_of(sizes, [](auto n) { return n == 0 || n > (16ull << 20); }))
            throw std::invalid_argument("lengths must contain 1 to 16MiB bytes each");
        PublisherLimits limits;
        limits.budget = cfg.value("budget", 64ull << 20);
        limits.max_sessions = 16;
        limits.allow_every = true;
        limits.gpu = gpu;
        Description d{Element::Bytes, {}, 16, "[('seed', '<u8'), ('bytes', '<u8')]"};
        d.max_payload_bytes = *std::ranges::max_element(sizes);
        d.application_text = Json{{"example", "opaque-gpu-v1"}, {"encoding", "constant-byte"},
                                  {"source", source}, {"lengths", sizes}}.dump();
        auto calibrate = cfg.value("kind", std::string("opaque")) == "calibration";
        auto shape = cfg.value("shape", std::vector<std::uint64_t>{129, 131});
        auto darks = cfg.value("dark_frames", 4u), flats = cfg.value("flat_frames", 4u);
        auto generations = cfg.value("calibrations", 2u);
        if(calibrate)
        {
            if(shape.size() != 2 || shape[0] < 1 || shape[1] < 1 || shape[0] > 4096 ||
               shape[1] > 4096 || shape[0] * shape[1] > (8ull << 20) ||
               darks < 1 || darks > 32 || flats < 1 || flats > 32 || generations < 1 || generations > 16)
                throw std::invalid_argument("calibration shape/count outside example limits");
            d = {Element::U16, shape, 24,
                 "[('kind', '<u8'), ('calibration_id', '<u8'), ('sample', '<u8')]"};
            d.application_text = Json{{"example", "dark-flat-v1"}, {"source", source},
                                      {"dark_frames", darks}, {"flat_frames", flats},
                                      {"calibrations", generations}, {"saturation", 65535},
                                      {"pattern", "spatial-u16-v1"}}.dump();
        }
        // Construct first so a refused configuration leaves the current publisher intact.
        auto next = std::make_unique<Publisher>(std::move(d), limits);
        frames = {};
        detach_publisher(*this);
        publisher = std::move(next);
        lengths = std::move(sizes);
        source_gpu = gpu;
        calibration = calibrate;
        image_pixels = calibrate ? shape[0] * shape[1] : 0;
        dark_frames = darks;
        flat_frames = flats;
        calibrations = generations;
        started = false;
        attach_publisher(*this, *publisher);
    }
    void start(Tango::DevLong64 count)
    {
        std::lock_guard lock(commands);
        if(count < 1 || started)
            throw std::invalid_argument("Start needs a positive count and a fresh Configure");
        publisher->ucx_start(); // every subscriptions and pullers must have joined already
        started = true;
        running = true;
        frames = std::jthread([this, count](std::stop_token stop)
        {
#ifdef EXAMPLES_CUDA
            cudaStream_t stream = nullptr;
#endif
            try
            {
#ifdef EXAMPLES_CUDA
                auto check = [](cudaError_t s)
                {
                    if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
                };
                if(source_gpu >= 0)
                {
                    check(cudaSetDevice(source_gpu));
                    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                }
#endif
                auto data_count = std::uint64_t(count);
                auto per_calibration = dark_frames + flat_frames + data_count;
                auto total = calibration ? calibrations * per_calibration : data_count;
                std::vector<std::uint16_t> image(image_pixels);
                for(std::uint64_t index = 0; index < total && !stop.stop_requested();)
                {
                    auto slot = publisher->acquire(Deadline::clock::now() + 100ms);
                    if(!slot)
                    {
                        if(!publisher->health().failure.empty())
                            throw std::runtime_error(publisher->health().failure);
                        continue;
                    }
                    if(calibration)
                    {
                        auto id = index / per_calibration + 1;
                        auto position = index % per_calibration;
                        std::uint64_t kind = position < dark_frames ? 0 : position < dark_frames + flat_frames ? 1 : 2;
                        auto sample = position - (kind == 0 ? 0 : kind == 1 ? dark_frames : dark_frames + flat_frames);
                        std::uint64_t fields[]{kind, id, sample};
                        std::memcpy(slot->fields().data(), fields, sizeof(fields));
                        for(std::uint64_t p = 0; p < image_pixels; ++p)
                            image[p] = calibration_pixel(p, kind, id, sample, kind == 0 ? dark_frames : flat_frames);
                        auto bytes = image.size() * sizeof(image[0]);
#ifdef EXAMPLES_CUDA
                        if(source_gpu >= 0)
                        {
                            check(cudaMemcpyAsync(slot->gpu_payload({stream}), image.data(), bytes,
                                                  cudaMemcpyHostToDevice, stream));
                            // This simulated source reuses one host generation buffer. Finish
                            // its upload before rewriting it; receive/analysis still overlap.
                            check(cudaStreamSynchronize(stream));
                        }
                        else
#endif
                            std::memcpy(slot->payload().data(), image.data(), bytes);
                        std::move(*slot).publish();
                        ++index;
                        continue;
                    }
                    std::uint64_t bytes = lengths[index % lengths.size()];
                    std::uint64_t seed = (index * 17 + 3) & 255;
                    std::uint64_t fields[]{seed, bytes};
                    std::memcpy(slot->fields().data(), fields, sizeof(fields));
#ifdef EXAMPLES_CUDA
                    if(source_gpu >= 0)
                    {
                        if(bytes > slot->payload_bytes()) throw std::runtime_error("slot capacity");
                        check(cudaMemsetAsync(slot->gpu_payload({stream}), int(seed), bytes, stream));
                    }
                    else
#endif
                        std::memset(slot->payload().data(), int(seed), bytes);
                    // No caller synchronization: readiness is recorded after queued GPU writes.
                    std::move(*slot).publish_bytes(bytes);
                    ++index;
                }
            }
            catch(const std::exception &e)
            {
                std::cerr << "publication failed: " << e.what() << std::endl;
            }
            publisher->finish();
#ifdef EXAMPLES_CUDA
            if(stream)
            {
                // The stream belongs to the example, so its destruction also waits for writes.
                (void)cudaStreamSynchronize(stream);
                (void)cudaStreamDestroy(stream);
            }
#endif
            running = false;
        });
    }
    std::mutex commands;
    std::unique_ptr<Publisher> publisher;
    std::vector<std::uint64_t> lengths;
    int source_gpu = -1;
    bool calibration = false;
    std::uint64_t image_pixels = 0;
    std::uint32_t dark_frames = 4, flat_frames = 4, calibrations = 2;
    bool started = false;
    std::atomic<bool> running{false};
    std::jthread frames;
};

class Command : public Tango::Command
{
  public:
    Command(const char *name, Tango::CmdArgType in) : Tango::Command(name, in, Tango::DEV_VOID) {}
    CORBA::Any *execute(Tango::DeviceImpl *impl, const CORBA::Any &in) override
    {
        auto &d = *static_cast<Device *>(impl);
        try
        {
            if(get_name() == "Configure")
            {
                Tango::DevString text = nullptr;
                extract(in, text);
                d.configure(text);
            }
            else
            {
                Tango::DevLong64 count = 0;
                extract(in, count);
                d.start(count);
            }
        }
        catch(const std::exception &e)
        {
            Tango::Except::throw_exception("ExampleRefused", e.what(), "opaque_publisher");
        }
        return insert();
    }
};
class DeviceClass : public Tango::DeviceClass
{
  public:
    DeviceClass() : Tango::DeviceClass(std::string("OpaqueGpuExample")) {}
    void command_factory() override
    {
        command_list.push_back(new Command("Configure", Tango::DEV_STRING));
        command_list.push_back(new Command("Start", Tango::DEV_LONG64));
        install_ucx_commands(command_list);
    }
    void attribute_factory(std::vector<Tango::Attr *> &list) override { install_ucx_attributes(list); }
    void device_factory(const Tango::DevVarStringArray *names) override
    {
        for(CORBA::ULong i = 0; i < names->length(); ++i)
            device_list.push_back(new Device(this, std::string((*names)[i])));
        for(auto *d : device_list) export_device(d, d->get_name().c_str());
    }
};
class Server : public Tango::DServer
{
  public:
    using Tango::DServer::DServer;
    void class_factory() override { add_class(new DeviceClass); }
};
} // namespace

int main(int argc, char **argv)
{
    try
    {
        auto *util = Tango::Util::init(argc, argv);
        util->register_dserver_constructor(
            [](Tango::DeviceClass *c, const std::string &name, const std::string &description,
               Tango::DevState state, const std::string &status) -> Tango::DServer *
            { return new Server(c, name.c_str(), description.c_str(), state, status.c_str()); });
        util->server_init();
        util->server_run();
        util->server_cleanup();
        return 0;
    }
    catch(const Tango::DevFailed &e) { Tango::Except::print_exception(e); }
    catch(const std::exception &e) { std::cerr << e.what() << std::endl; }
    return 1;
}
