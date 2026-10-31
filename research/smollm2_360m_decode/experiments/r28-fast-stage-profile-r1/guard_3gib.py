"""Resource guard for decode baseline; canonical disk root is prime-rl."""
import json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

repo = Path("/home/pradheep/tiny-prl/prime-rl")
artifact = Path(sys.argv[1]); artifact.mkdir(parents=True, exist_ok=True)
command = sys.argv[2:]
floor = 2 * 1024**3
disk_before = shutil.disk_usage(repo).free
if disk_before < floor:
    result = {"command": command, "returncode": None, "stop_reason": "prime-rl disk free below 2 GiB", "elapsed_seconds": 0, "minimum_available_bytes": None, "peak_cgroup_bytes": 0, "peak_gpu_mib": 0, "disk_free_before_bytes": disk_before, "minimum_disk_free_bytes": disk_before}
    (artifact / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result)); sys.exit(1)
stop = False
def request_stop(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
cg = Path("/sys/fs/cgroup") / cgroup.lstrip("/"); current_path = cg / "memory.current"
started = time.monotonic(); min_ram = float("inf"); peak_cg = peak_gpu = peak_swap = 0; min_disk = disk_before; reason = None
with (artifact / "launch.log").open("w") as log, (artifact / "gpu.csv").open("w") as gpu, (artifact / "host.jsonl").open("w") as host:
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu", "--format=csv", "--loop-ms=200"], stdout=gpu, stderr=subprocess.DEVNULL)
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while proc.poll() is None:
            mem = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
            available = mem["MemAvailable"]; disk_free = shutil.disk_usage(repo).free; min_disk = min(min_disk, disk_free)
            current = int(current_path.read_text()); swap = int((cg / "memory.swap.current").read_text())
            peak_swap = max(peak_swap, swap)
            min_ram = min(min_ram, available); peak_cg = max(peak_cg, current)
            host.write(json.dumps({"elapsed": time.monotonic()-started, "available_bytes": available, "disk_free_bytes": disk_free, "cgroup_bytes": current, "swap_bytes": swap})+"\n"); host.flush()
            lines = (artifact / "gpu.csv").read_text().splitlines()
            if lines:
                fields = lines[-1].split(",")
                if len(fields) == 4 and fields[1].strip().endswith(" MiB"):
                    peak_gpu = max(peak_gpu, int(fields[1].strip().split()[0]))
            if stop: reason = "termination requested"
            elif available < floor: reason = "available system RAM below 2 GiB"
            elif disk_free < floor: reason = "prime-rl disk free below 2 GiB"
            elif peak_gpu > 3072: reason = "GPU memory used exceeded 3072 MiB"
            elif time.monotonic()-started > 600: reason = "600 second deadline"
            if reason:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            time.sleep(.1)
    finally:
        if proc.poll() is None: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=15)
        sampler.terminate(); sampler.wait(timeout=10)
result = {"command": command, "returncode": proc.returncode, "stop_reason": reason, "elapsed_seconds": time.monotonic()-started, "minimum_available_bytes": min_ram, "peak_cgroup_bytes": peak_cg, "peak_swap_bytes": peak_swap, "peak_gpu_mib": peak_gpu, "disk_free_before_bytes": disk_before, "minimum_disk_free_bytes": min_disk}
(artifact / "result.json").write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result))
sys.exit(0 if proc.returncode == 0 and reason is None else 1)
