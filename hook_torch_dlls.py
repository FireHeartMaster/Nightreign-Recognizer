import os
import sys

# PyInstaller on Windows may bundle torch DLLs under _internal/torch/lib.
# Ensure that directory is available to Windows dynamic loader before torch is imported.
if os.name == "nt":
    debug_log = os.path.join(os.path.dirname(sys.executable), "hook_debug.log")
    
    base = getattr(sys, "_MEIPASS", None)
    if base is not None and os.path.basename(base).lower() == "_internal":
        base = os.path.dirname(base)
    if base is None or not os.path.isdir(base):
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        if os.path.basename(base).lower() == "_internal":
            base = os.path.dirname(base)
        if not os.path.isdir(base):
            base = os.path.dirname(sys.executable)
            if os.path.basename(base).lower() == "_internal":
                base = os.path.dirname(base)
    torch_lib_dir = os.path.join(base, "_internal", "torch", "lib")
    debug_lines = []
    debug_lines.append(f"hook_torch_dlls sys_executable={sys.executable}")
    debug_lines.append(f"hook_torch_dlls sys_argv0={sys.argv[0]}")
    debug_lines.append(f"hook_torch_dlls base={base}")
    debug_lines.append(f"hook_torch_dlls torch_lib_dir={torch_lib_dir}")
    if os.path.isdir(torch_lib_dir):
        try:
            os.add_dll_directory(torch_lib_dir)
            debug_lines.append("hook_torch_dlls add_dll_directory ok")
        except Exception as exc:
            debug_lines.append(f"hook_torch_dlls add_dll_directory failed: {exc}")
        os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            import ctypes

            ctypes.windll.kernel32.SetDllDirectoryW(torch_lib_dir)
            debug_lines.append("hook_torch_dlls SetDllDirectoryW ok")
        except Exception as exc:
            debug_lines.append(f"hook_torch_dlls SetDllDirectoryW failed: {exc}")
        try:
            import ctypes
            for dll_name in ["c10.dll", "torch.dll", "torch_python.dll"]:
                dll_path = os.path.join(torch_lib_dir, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.WinDLL(dll_path)
                        debug_lines.append(f"hook_torch_dlls loaded {dll_name}")
                    except Exception as exc:
                        debug_lines.append(f"hook_torch_dlls failed to load {dll_name}: {exc}")
                else:
                    debug_lines.append(f"hook_torch_dlls missing {dll_name}")
        except Exception as exc:
            debug_lines.append(f"hook_torch_dlls ctypes test failed: {exc}")
    else:
        debug_lines.append("hook_torch_dlls torch_lib_dir missing")
    try:
        print("\n".join(debug_lines))
    except Exception:
        pass
    try:
        log_path = os.path.join(base, "hook_torch_dlls.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(debug_lines))
    except Exception:
        pass
