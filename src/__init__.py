"""
Root package for Temperature Prediction Cybersecurity Extension.
Ensures runtime compatibility across Windows MSVC and TensorFlow environments.
"""

import os
import sys
import ctypes

# Fix for Windows: ensure modern System32 C++ runtime is loaded before Anaconda's older bundled DLLs
if sys.platform == "win32":
    for dll_name in [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
    ]:
        sys32_path = os.path.join(r"C:\Windows\System32", dll_name)
        if os.path.exists(sys32_path):
            try:
                ctypes.CDLL(sys32_path)
            except Exception:
                pass

    # Suppress verbose oneDNN notification messages
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

