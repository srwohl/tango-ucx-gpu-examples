"""Select the existing persistent FBP shim independently of scan overlap."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get("TANGO_SCAN_BACKEND") == "persistent":
    import fbp_shim.sitecustomize

