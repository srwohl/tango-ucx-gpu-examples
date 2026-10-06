// CPU-only regression of the settings state used by the Tango device.
#include "reconstruction_settings.h"
#include <cassert>
#include <iostream>

template<class Operation> void refused(Operation operation) {
    bool rejected = false;
    try { operation(); } catch(const std::runtime_error &) { rejected = true; }
    assert(rejected);
}

int main() {
    using Json = nlohmann::json;
    const Json initial{{"algorithm", "gridrec"}, {"scale_factor", 1}};
    const Json fbp{{"algorithm", "fbp"}, {"scale_factor", 2}};
    const Json sirt{{"algorithm", "sirt"}, {"iterations", 12}};
    ReconstructionSettings settings(initial);
    assert(settings.state(64).at("active").is_null());
    refused([&] { settings.for_scan(100); });
    assert(settings.begin_scan(100) == initial);
    settings.configure(fbp);
    settings.configure(fbp);  // Idempotent writes do not create extra revisions.
    assert(settings.state(64).at("requested").at("revision") == 1);
    assert(settings.for_scan(100).at("options") == initial); // Current scan stays immutable.
    settings.configure(sirt); // Last queued update wins at the next boundary.
    assert(settings.begin_scan(101) == sirt);
    assert(settings.for_scan(101).at("revision") == 2);
    assert(settings.begin_scan(102) == sirt);
    settings.configure(fbp);
    assert(settings.begin_scan(103) == fbp);
    // A delayed reader resolves earlier settings after multiple subsequent changes.
    assert(settings.for_scan(100).at("options") == initial);
    assert(settings.for_scan(101).at("options") == sirt);
    assert(settings.for_scan(102).at("scan_id") == 102);
    assert(settings.for_scan(102).at("revision") == 2);
    assert(settings.for_scan(103).at("options") == fbp);
    refused([&] { settings.for_scan(99); });
    refused([&] { settings.for_scan(104); });
    refused([&] { settings.begin_scan(103); });
    settings.finish();
    refused([&] { settings.configure(initial); });
    assert(settings.for_scan(100).at("options") == initial);
    assert(settings.state(64).at("finished") == true);
    std::cout << "Reconstruction settings scan-boundary checks passed\n";
}
