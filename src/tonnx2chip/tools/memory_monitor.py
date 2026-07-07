"""Host & NPU memory monitoring utility."""

import threading
import time


class MemoryMonitor:
    """Background thread that samples host RSS and NPU memory usage."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_host_mb = 0.0
        self.peak_npu_mb = 0.0

    def _sample(self):
        try:
            import psutil
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            self.peak_host_mb = max(self.peak_host_mb, rss_mb)
        except Exception:
            pass

        try:
            import subprocess
            result = subprocess.run(
                ["npu-smi", "info", "-t", "memory", "-o", "csv"],
                capture_output=True, text=True, timeout=2,
            )
            for line in result.stdout.splitlines():
                if "HBM" in line or "Used" in line:
                    parts = line.split(",")
                    for p in parts:
                        p = p.strip().rstrip("%MB").rstrip("MB")
                        try:
                            val = float(p)
                            if val > 0:
                                self.peak_npu_mb = max(self.peak_npu_mb, val)
                                break
                        except ValueError:
                            continue
        except Exception:
            pass

    def _run(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def summary(self) -> str:
        return (
            f"Peak Host RSS: {self.peak_host_mb:.1f} MB | "
            f"Peak NPU Mem: {self.peak_npu_mb:.1f} MB"
        )
