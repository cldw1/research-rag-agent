"""Process-local native library setup; never changes the user's system PATH."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from threading import RLock


_dll_handles = []  # Keep handles alive: dropping add_dll_directory's result closes it.
_native_handles = []
_lock = RLock()
_prepared = False


def load_faiss():
    global _prepared
    with _lock:
        if os.name == "nt" and not _prepared:
            locations = [Path(sys.base_prefix) / "Library" / "bin"]
            spec = importlib.util.find_spec("faiss")
            if spec and spec.origin:
                locations.append(Path(spec.origin).parent.parent / "faiss_cpu.libs")
            for location in locations:
                if location.is_dir():
                    _dll_handles.append(os.add_dll_directory(str(location)))
            # Torch and FAISS share native runtime dependencies on Windows.
            # Initialize the already-required Torch runtime in a consistent order.
            importlib.import_module("torch")
            # Conda and the FAISS wheel can both ship DLLs with the same basename.
            # Load the wheel's own dependencies by absolute path before its SWIG
            # module, rather than relying on unspecified directory search order.
            if spec and spec.origin:
                import ctypes
                bundled = Path(spec.origin).parent.parent / "faiss_cpu.libs"
                for name in ("msvcp140.dll", "vcomp140.dll", "libopenblas.dll"):
                    dependency = bundled / name
                    if dependency.is_file():
                        _native_handles.append(ctypes.WinDLL(str(dependency), winmode=0x00001100))
            _prepared = True
        return importlib.import_module("faiss")
