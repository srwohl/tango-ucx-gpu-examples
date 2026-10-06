// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>

// A spatially varying detector response. Calibration noise is symmetric over each stack.
inline std::uint16_t calibration_pixel(std::uint64_t pixel, std::uint64_t kind,
                                       std::uint64_t id, std::uint64_t sample,
                                       std::uint64_t samples)
{
    auto dark = std::int64_t(100 + pixel % 31 + id * 3);
    auto response = std::int64_t(1 + (pixel + id) % 4);
    auto noise = 2 * std::int64_t(sample) - std::int64_t(samples - 1);
    if(kind == 0)
        return std::uint16_t(dark + noise);
    if(kind == 1)
    {
        auto illumination = pixel % 97 == 0 ? 0 : pixel % 193 == 0 ? -16 : response * 4096;
        return std::uint16_t(dark + illumination + noise);
    }
    if(pixel % 251 == 0)
        return 65535; // saturated data
    if(pixel % 389 == 0)
        return std::uint16_t(dark - 5); // retain negative dark-subtracted values
    return std::uint16_t(dark + response * std::int64_t((sample * 7 + pixel * 3 + id * 13) % 3072));
}
