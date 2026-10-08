#include <tango-ucx/ucx.h>
#include <transport.h>
#include <wire.h>
#include <ucp/api/ucp.h>
#include "fixtures/fork.h"
#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace TangoUcx;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

namespace
{
std::array<std::atomic<std::uint64_t>, 4> sends{};
std::array<std::atomic<std::uint64_t>, 4> send_ns{};
bool time_calls = false;
std::uint64_t elapsed_ns(Clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
Deadline deadline()
{
    return Clock::now() + std::chrono::seconds(30);
}
void send_text(int fd, std::string_view document)
{
    if(!test::write_text(fd, document))
        throw std::runtime_error("pipe write failed");
}
Json counts()
{
    Json result;
    const char *names[] = {"ready", "frame", "credit", "end"};
    for(std::size_t index = 0; index < sends.size(); ++index)
        result[names[index]] = {{"calls", sends[index].load()}, {"host_ns", send_ns[index].load()}};
    return result;
}
std::size_t frame_length(std::uint64_t index, std::size_t bytes, bool varying)
{
    return varying ? 16 + ((index + 1) * 2654435761ull % (bytes - 15)) : bytes;
}
Json header_probe(std::uint64_t iterations, unsigned field_bytes)
{
    wire::FrameMessage message{};
    message.session = 7;
    message.payload_bytes = 75000;
    message.record_bytes = field_bytes;
    message.packed = true;
    std::array<std::byte, 1072> header{};
    wire::FrameMessage decoded{};
    std::uint64_t checksum = 0;
    auto start = Clock::now();
    for(std::uint64_t index = 0; index < iterations; ++index)
    {
        message.sequence = index;
        message.position = index * message.payload_bytes;
        auto bytes = wire::encode_into(header, message);
        if(!wire::decode({header.data(), bytes}, field_bytes, decoded))
            throw std::runtime_error("header decode failed");
        checksum += decoded.sequence + decoded.position;
    }
    auto duration = elapsed_ns(start);
    return {{"iterations", iterations}, {"field_bytes", field_bytes},
            {"header_bytes", 48 + field_bytes}, {"encode_decode_ns_per_frame", double(duration) / iterations},
            {"checksum", checksum}};
}
struct RawWorker
{
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
    RawWorker()
    {
        ucp_config_t *config = nullptr;
        if(ucp_config_read(nullptr, nullptr, &config) != UCS_OK)
            throw std::runtime_error("raw config failed");
        ucp_params_t context_params{};
        context_params.field_mask = UCP_PARAM_FIELD_FEATURES;
        context_params.features = UCP_FEATURE_AM;
        auto status = ucp_init(&context_params, config, &context);
        ucp_config_release(config);
        if(status != UCS_OK)
            throw std::runtime_error("raw context failed");
        ucp_worker_params_t worker_params{};
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
        if(ucp_worker_create(context, &worker_params, &worker) != UCS_OK)
            throw std::runtime_error("raw worker failed");
    }
    ~RawWorker()
    {
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
    }
};
struct RawReceived
{
    bool arrived = false, valid = false, rendezvous = false;
    std::size_t expected = 0, bytes = 0, header_bytes = 0;
    std::uint64_t first_header_word = 0;
    static ucs_status_t on_frame(void *argument, const void *header, size_t header_length,
                                 void *data, size_t length, const ucp_am_recv_param_t *params)
    {
        auto &received = *static_cast<RawReceived *>(argument);
        received.rendezvous = params->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;
        received.bytes = length;
        received.header_bytes = header_length;
        if(header_length >= 8)
            std::memcpy(&received.first_header_word, header, 8);
        received.valid = !received.rendezvous && length == received.expected && header_length == 88;
        if(received.valid)
        {
            auto *header_bytes = static_cast<const unsigned char *>(header);
            auto *payload_bytes = static_cast<const unsigned char *>(data);
            for(std::size_t offset = 0; offset < header_length; ++offset)
                received.valid &= header_bytes[offset] == 0xa5;
            for(std::size_t offset = 0; offset < length; ++offset)
                received.valid &= payload_bytes[offset] == 0x5a;
        }
        received.arrived = true;
        return received.rendezvous ? UCS_ERR_REJECTED : UCS_OK;
    }
};
Json raw_probe(std::size_t bytes)
{
    test::Forked peer;
    if(peer.pid == 0)
    {
        RawWorker worker;
        RawReceived received;
        received.expected = bytes;
        ucp_am_handler_param_t handler{};
        handler.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                             UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
        handler.id = 1;
        handler.flags = UCP_AM_FLAG_WHOLE_MSG;
        handler.cb = RawReceived::on_frame;
        handler.arg = &received;
        if(ucp_worker_set_am_recv_handler(worker.worker, &handler) != UCS_OK)
            _exit(3);
        ucp_address_t *address = nullptr;
        std::size_t address_length = 0;
        ucp_worker_get_address(worker.worker, &address, &address_length);
        send_text(peer.out(), {reinterpret_cast<const char *>(address), address_length});
        ucp_worker_release_address(worker.worker, address);
        auto until = deadline();
        while(!received.arrived && Clock::now() < until)
            ucp_worker_progress(worker.worker);
        Json result = {{"raw_ucx", true}, {"expected_bytes", bytes}, {"received_bytes", received.bytes},
                       {"header_bytes", received.header_bytes}, {"first_header_word", received.first_header_word},
                       {"arrived", received.arrived}, {"valid", received.valid},
                       {"rendezvous", received.rendezvous}};
        send_text(peer.out(), result.dump());
        test::read_text(peer.in());
        _exit(received.valid ? 0 : 2);
    }
    RawWorker worker;
    auto address = test::read_text(peer.in());
    ucp_ep_params_t endpoint_params{};
    endpoint_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    endpoint_params.address = reinterpret_cast<const ucp_address_t *>(address.data());
    ucp_ep_h endpoint = nullptr;
    if(ucp_ep_create(worker.worker, &endpoint_params, &endpoint) != UCS_OK)
        throw std::runtime_error("raw endpoint failed");
    std::array<unsigned char, 88> header;
    header.fill(0xa5);
    std::vector<unsigned char> payload(bytes, 0x5a);
    ucp_request_param_t params{};
    params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    params.flags = UCP_AM_SEND_FLAG_EAGER;
    auto *request = ucp_am_send_nbx(endpoint, 1, header.data(), header.size(), payload.data(), bytes, &params);
    if(UCS_PTR_IS_PTR(request))
    {
        auto until = deadline();
        while(ucp_request_check_status(request) == UCS_INPROGRESS && Clock::now() < until)
            ucp_worker_progress(worker.worker);
        if(ucp_request_check_status(request) == UCS_INPROGRESS)
            throw std::runtime_error("raw send timeout");
        ucp_request_free(request);
    }
    else if(UCS_PTR_IS_ERR(request))
        throw std::runtime_error("raw send failed");
    auto result = Json::parse(test::read_text(peer.in()));
    send_text(peer.out(), "exit");
    ucp_ep_destroy(endpoint);
    return result;
}
Json transfer_probe(std::uint64_t frames, std::size_t bytes, unsigned batch_size,
                    unsigned field_bytes, unsigned wait_us, bool varying, bool validate_all)
{
    test::Forked peer;
    if(peer.pid == 0)
    {
        try
        {
            auto offer = transport::parse_ucx_stream(test::read_text(peer.in()));
            SubscriptionOptions options;
            options.budget = 32ull << 20;
            options.batch = batch_size;
            options.max_wait = std::chrono::microseconds(wait_us);
            auto start = Clock::now();
            auto pending = transport::prepare(offer, true, options);
            auto prepare_ns = elapsed_ns(start);
            send_text(peer.out(), pending.open_message());
            auto reply = test::read_text(peer.in());
            start = Clock::now();
            auto sub = std::move(pending).start(reply, deadline(), {});
            auto handshake_ns = elapsed_ns(start);
            send_text(peer.out(), "ready");
            std::uint64_t received = 0, batches = 0, received_bytes = 0;
            bool valid = true;
            start = Clock::now();
            while(auto batch = sub.read(deadline()))
            {
                ++batches;
                auto *records = static_cast<const std::byte *>(batch->records());
                for(std::size_t frame = 0; frame < batch->frames(); ++frame, ++received)
                {
                    auto payload = batch->payload(frame);
                    auto expected_bytes = frame_length(received, bytes, varying);
                    valid &= batch->index(frame) == received && payload.size() == expected_bytes;
                    valid &= wire::get64(payload.data()) == received;
                    valid &= wire::get64(payload.data() + payload.size() - 8) == received;
                    if(field_bytes)
                        valid &= wire::get64(records + frame * (16 + field_bytes) + 16) == received;
                    if(validate_all)
                        for(std::size_t offset = 8; offset + 8 < payload.size(); ++offset)
                            valid &= payload[offset] == std::byte(received & 255);
                    received_bytes += payload.size();
                }
            }
            auto receive_ns = elapsed_ns(start);
            Json result = {{"received", received}, {"batches", batches}, {"payload_bytes", received_bytes},
                           {"receive_ns", receive_ns}, {"prepare_ns", prepare_ns},
                           {"native_ready_ns", handshake_ns}, {"valid", valid && received == frames},
                           {"outcome", sub.outcome() ? int(*sub.outcome()) : 0},
                           {"transport", sub.health().transport}, {"failure", sub.health().failure},
                           {"ucx_calls", counts()}};
            send_text(peer.out(), result.dump());
            test::read_text(peer.in());
            sub.close(deadline());
            _exit(result["valid"].get<bool>() && sub.outcome() == Outcome::End ? 0 : 2);
        }
        catch(const std::exception &error)
        {
            send_text(peer.out(), Json{{"error", error.what()}}.dump());
            _exit(2);
        }
    }
    Description description{Element::Bytes, {}, field_bytes, field_bytes ? "u1" : ""};
    description.max_payload_bytes = bytes;
    PublisherLimits limits;
    limits.budget = 32ull << 20;
    limits.max_sessions = 1;
    limits.allow_every = true;
    auto start = Clock::now();
    Publisher publisher(description, limits);
    auto publisher_create_ns = elapsed_ns(start);
    send_text(peer.out(), transport::ucx_stream(publisher));
    auto open = test::read_text(peer.in());
    start = Clock::now();
    auto reply = transport::handle(publisher, open);
    auto open_handle_ns = elapsed_ns(start);
    send_text(peer.out(), reply);
    if(test::read_text(peer.in()) != "ready")
        throw std::runtime_error("peer did not become ready");
    publisher.ucx_start();
    start = Clock::now();
    std::uint64_t acquire_ns = 0;
    for(std::uint64_t index = 0; index < frames; ++index)
    {
        auto acquire_start = Clock::now();
        auto slot = publisher.acquire(deadline());
        acquire_ns += elapsed_ns(acquire_start);
        if(!slot)
            break;
        auto length = frame_length(index, bytes, varying);
        auto payload = slot->payload();
        std::memset(payload.data(), int(index & 255), length);
        std::memcpy(payload.data(), &index, 8);
        std::memcpy(payload.data() + length - 8, &index, 8);
        if(field_bytes)
        {
            std::fill(slot->fields().begin(), slot->fields().end(), std::byte{0});
            std::memcpy(slot->fields().data(), &index, 8);
        }
        std::move(*slot).publish_bytes(length, index);
    }
    publisher.finish();
    auto receiver = Json::parse(test::read_text(peer.in()));
    auto transfer_ns = elapsed_ns(start);
    auto result = Json{{"frames", frames}, {"max_payload_bytes", bytes}, {"batch", batch_size},
                       {"field_bytes", field_bytes}, {"max_wait_us", wait_us}, {"varying_lengths", varying},
                       {"validate_all", validate_all}, {"time_ucx_calls", time_calls},
                       {"publisher_create_ns", publisher_create_ns}, {"open_handle_ns", open_handle_ns},
                       {"transfer_ns", transfer_ns}, {"acquire_ns", acquire_ns},
                       {"ns_per_frame", double(transfer_ns) / frames},
                       {"ucx_calls", counts()}, {"receiver", receiver}};
    send_text(peer.out(), "exit");
    return result;
}
}

extern "C" ucs_status_ptr_t ucp_am_send_nbx(ucp_ep_h endpoint, unsigned id, const void *header,
                                            size_t header_length, const void *buffer, size_t count,
                                            const ucp_request_param_t *params)
{
    using Send = decltype(&ucp_am_send_nbx);
    static auto original = reinterpret_cast<Send>(dlsym(RTLD_NEXT, "ucp_am_send_nbx"));
    if(!original)
        std::abort();
    auto start = time_calls ? Clock::now() : Clock::time_point{};
    auto result = original(endpoint, id, header, header_length, buffer, count, params);
    if(id < sends.size())
    {
        sends[id].fetch_add(1, std::memory_order_relaxed);
        if(time_calls)
            send_ns[id].fetch_add(elapsed_ns(start), std::memory_order_relaxed);
    }
    return result;
}

int main(int argc, char **argv)
{
    try
    {
        std::uint64_t frames = 10000;
        std::size_t bytes = 75000;
        unsigned batch = 16, fields = 40, wait_us = 10000;
        bool header = false, raw = false, varying = false, validate_all = false;
        for(int index = 1; index < argc; ++index)
        {
            std::string argument = argv[index];
            if(argument == "--header")
                header = true;
            else if(argument == "--raw")
                raw = true;
            else if(argument == "--varying")
                varying = true;
            else if(argument == "--validate-all")
                validate_all = true;
            else if(argument == "--time-ucx")
                time_calls = true;
            else if(index + 1 < argc && argument == "--frames")
                frames = std::stoull(argv[++index]);
            else if(index + 1 < argc && argument == "--bytes")
                bytes = std::stoull(argv[++index]);
            else if(index + 1 < argc && argument == "--batch")
                batch = std::stoul(argv[++index]);
            else if(index + 1 < argc && argument == "--fields")
                fields = std::stoul(argv[++index]);
            else if(index + 1 < argc && argument == "--wait-us")
                wait_us = std::stoul(argv[++index]);
            else
                throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
        if(!frames || bytes < 16 || fields > 1024 || fields % 8 || !batch || batch > 1024)
            throw std::invalid_argument("invalid probe dimensions");
        auto result = raw ? raw_probe(bytes)
                          : header ? header_probe(frames, fields)
                                   : transfer_probe(frames, bytes, batch, fields, wait_us, varying, validate_all);
        std::cout << result.dump() << '\n';
        if(raw)
            return result.value("valid", false) ? 0 : 2;
        if(!header && (!result["receiver"].value("valid", false) || result["receiver"].value("outcome", 0) != 1))
            return 2;
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
