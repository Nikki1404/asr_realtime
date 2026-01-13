import time, threading
from app.metrics import GPU_UTIL, GPU_MEM_USED_MB, GPU_MEM_TOTAL_MB

def start_gpu_monitor(enable, gpu_index):
    if not enable:
        return
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def loop():
        while True:
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            GPU_UTIL.set(util)
            GPU_MEM_USED_MB.set(mem.used / 1e6)
            GPU_MEM_TOTAL_MB.set(mem.total / 1e6)
            time.sleep(2)

    threading.Thread(target=loop, daemon=True).start()
