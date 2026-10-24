"""Run the complete reproducible audit, matching, verification, and report."""

import ctypes
from datetime import datetime, timezone
import platform
import time

import audit
import match
import report
import verify
from common import CACHE, save_json


def peak_working_set():
    if platform.system() != "Windows":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def run():
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    audit.run()
    match.run()
    verify.run()
    save_json("run_manifest.json", {
        "started_at_utc": started_at,
        "finished_analysis_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_wall_seconds": round(time.perf_counter() - started, 3),
        "python_version": platform.python_version(), "platform": platform.system(),
        "peak_process_working_set_bytes": peak_working_set(),
        "duckdb_memory_budget": "768MB", "duckdb_threads": 2,
        "cache_database_bytes": (CACHE / "audit.duckdb").stat().st_size,
        "cache_directory": str(CACHE),
    })
    report.run()


if __name__ == "__main__":
    run()
