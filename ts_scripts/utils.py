import os
import platform
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

nvidia_smi_cmd = {
    "Windows": "nvidia-smi.exe",
    "Darwin": "nvidia-smi",
    "Linux": "nvidia-smi",
}


def is_gpu_instance():
    return os.system(nvidia_smi_cmd[platform.system()]) == 0


def is_conda_build_env():
    return os.system("conda-build") == 0


def is_conda_env():
    return os.system("conda") == 0


def check_python_version():
    cur_version = sys.version_info

    req_version = (3, 8)
    if (
        cur_version.major != req_version[0]
        or cur_version.minor < req_version[1]
    ):
        print(f"System version{str(cur_version)}")
        print(
            f"TorchServe supports Python {req_version[0]}.{req_version[1]} and higher only. Please upgrade"
        )
        exit(1)


def check_ts_version():
    from ts.version import __version__

    return __version__


def try_and_handle(cmd, dry_run=False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)
