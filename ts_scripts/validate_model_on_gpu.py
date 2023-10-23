import nvgpu

gpu_info = nvgpu.gpu_info()

model_loaded = any(
    info['mem_used'] > 0 and info['mem_used_percent'] > 0.0
    for info in gpu_info
)
if not model_loaded:
    exit(1)
