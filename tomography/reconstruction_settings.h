// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>

// Caller serializes access. Options have already passed scientific validation.
class ReconstructionSettings {
    using Json = nlohmann::json;
public:
    explicit ReconstructionSettings(Json options) : requested(std::move(options)) {}

    void configure(const Json &options) {
        if(finished) throw std::runtime_error("reconstruction has finished; no more scans will use new settings");
        if(options != requested) {
            requested = options;
            ++revision;
        }
    }
    Json begin_scan(std::uint64_t scan_id) {
        if(finished) throw std::runtime_error("reconstruction has finished");
        if(!active.is_null() && scan_id <= active.at("scan_id").get<std::uint64_t>())
            throw std::runtime_error("reconstruction scan identity did not advance");
        const bool changed = active.is_null() || active.at("revision") != revision;
        active = Json{{"options", requested}, {"revision", revision}, {"scan_id", scan_id}};
        // Keep transitions, rather than one entry per volume, for delayed readers.
        if(changed) history.emplace(scan_id, active);
        return requested;
    }
    Json for_scan(std::uint64_t scan_id) const {
        if(active.is_null() || scan_id > active.at("scan_id").get<std::uint64_t>())
            throw std::runtime_error("reconstruction scan has not started");
        auto found = history.upper_bound(scan_id);
        if(found == history.begin()) throw std::runtime_error("unknown reconstruction scan");
        auto result = std::prev(found)->second;
        result["scan_id"] = scan_id;
        return result;
    }
    void finish() { finished = true; }
    Json state(std::uint64_t columns) const {
        return Json{{"requested", {{"options", requested}, {"revision", revision}}},
                    {"active", active}, {"finished", finished}, {"detector_columns", columns}};
    }
private:
    Json requested, active;
    std::uint64_t revision = 0;
    bool finished = false;
    std::map<std::uint64_t, Json> history;
};
