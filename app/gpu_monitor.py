import time
import threading
import logging
from app.metrics import GPU_UTIL, GPU_MEM_USED_MB, GPU_MEM_TOTAL_MB

log = logging.getLogger("asr_server")

def start_gpu_monitor(enable: bool, gpu_index: int):
    if not enable:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))

        def loop():
            while True:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    GPU_UTIL.set(util)
                    GPU_MEM_USED_MB.set(mem.used / (1024 * 1024))
                    GPU_MEM_TOTAL_MB.set(mem.total / (1024 * 1024))
                except Exception:
                    pass
                time.sleep(2)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        log.info("GPU monitor started")
    except Exception as e:
        log.warning(f"GPU monitor disabled: {e}")
