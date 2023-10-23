import argparse
import datetime
import os
import shutil
from subprocess import Popen

import ruamel.yaml
from utils import gen_md_report, gen_metrics_json, gen_model_config_json

CWD = os.getcwd()
MODEL_JSON_CONFIG_PATH = f"{CWD}/model_json_config"
BENCHMARK_TMP_PATH = "/tmp/benchmark"
BENCHMARK_REPORT_PATH = "/tmp/ts_benchmark"
TS_LOGS_PATH = f"{CWD}/logs"
MODEL_STORE = "/tmp/model_store"
WF_STORE = "/tmp/wf_store"


class BenchmarkConfig:
    def __init__(self, yaml_dict, skip_ts_install, skip_upload):
        self.yaml_dict = yaml_dict
        self.skip_ts_install = skip_ts_install
        self.skip_upload = skip_upload
        self.bm_config = {}
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        self.bm_config[
            "version"
        ] = f"torchserve-nightly=={yesterday.year}.{yesterday.month}.{yesterday.day}"
        self.bm_config["hardware"] = "cpu"

    def ts_version(self, version):
        for k, v in version.items():
            if k == "branch":
                self.bm_config["version"] = v
            elif k == "nightly":
                self.bm_config["version"] = f"torchserve-nightly=={v}"
            elif k == "release":
                self.bm_config["version"] = f"torchserve=={v}"
            break

    def models(self, model_files):
        self.bm_config["models"] = model_files

    def hardware(self, hw):
        self.bm_config["hardware"] = hw

    def metrics_cmd(self, cmd):
        cmd_options = []
        for key_value in cmd:
            for k, v in key_value.items():
                if k == "--namespace":
                    cmd_options.extend((k, "".join(v)))
                elif k == "cmd":
                    cmd_options.append(v)
                else:
                    cmd_options.extend((k, v))
                break

        self.bm_config["metrics_cmd"] = " ".join(cmd_options)

    def report_cmd(self, cmd):
        cmd_options = []
        for key_value in cmd:
            for k, v in key_value.items():
                if k == "dest":
                    for i in range(len(v)):
                        if v[i] == "today()":
                            today = datetime.date.today()
                            v[i] = f"{today.year}-{today.month}-{today.day}"
                            break
                    cmd_options.append(f'{"/".join(v)}/{self.bm_config["version"]}')
                else:
                    cmd_options.append(v)
                break

        self.bm_config["report_cmd"] = " ".join(cmd_options)

    def load_config(self):
        report_cmd = None
        for k, v in self.yaml_dict.items():
            if k == "ts_version":
                self.ts_version(v)
            elif k == "models":
                self.models(v)
            elif k == "hardware":
                self.hardware(v)
            elif k == "metrics_cmd" and not self.skip_upload:
                self.metrics_cmd(v)
            elif k == "report_cmd" and not self.skip_upload:
                report_cmd = v

        self.bm_config["model_config_path"] = (
            f'{MODEL_JSON_CONFIG_PATH}/{self.bm_config["hardware"]}'
            if self.bm_config["hardware"] in ["cpu", "gpu", "neuron", "neuronx"]
            else f"{MODEL_JSON_CONFIG_PATH}/cpu"
        )

        if self.skip_ts_install:
            self.bm_config["version"] = get_torchserve_version()

        if report_cmd:
            self.report_cmd(report_cmd)

        for k, v in self.bm_config.items():
            print(f"{k}={v}")


def load_benchmark_config(bm_config_path, skip_ts_install, skip_upload):
    yaml = ruamel.yaml.YAML()
    with open(bm_config_path, "r") as f:
        yaml_dict = yaml.load(f)

        benchmark_config = BenchmarkConfig(yaml_dict, skip_ts_install, skip_upload)
        benchmark_config.load_config()

    return benchmark_config.bm_config


def benchmark_env_setup(bm_config, skip_ts_install):
    install_torchserve(skip_ts_install, bm_config["hardware"], bm_config["version"])
    setup_benchmark_path(bm_config["model_config_path"])
    build_model_json_config(bm_config["models"])
    enable_launcher_with_logical_core(bm_config["hardware"])


def install_torchserve(skip_ts_install, hw, ts_version):
    if skip_ts_install:
        return

    # git checkout branch if it is needed
    cmd = "git checkout master && git reset --hard && git clean -dffx . && git pull --rebase"
    execute(cmd, wait=True)
    print("successfully reset git")

    ts_install_cmd = None
    if ts_version.startswith("torchserve==") or ts_version.startswith(
        "torchserve-nightly=="
    ):
        ts_install_cmd = f"pip install {ts_version}"
    else:
        cmd = f"git checkout {ts_version}"
        execute(cmd, wait=True)

    # install_dependencies.py
    if hw == "gpu":
        cmd = "python ts_scripts/install_dependencies.py --environment dev --cuda cu117"
    else:
        cmd = "python ts_scripts/install_dependencies.py --environment dev"
    execute(cmd, wait=True)
    print("successfully install install_dependencies.py")

    # install torchserve
    if ts_install_cmd is None:
        ts_install_cmd = "python ts_scripts/install_from_src.py"
    execute(ts_install_cmd, wait=True)
    print("successfully install torchserve")


def setup_benchmark_path(model_config_path):
    benchmark_path_list = [BENCHMARK_TMP_PATH, BENCHMARK_REPORT_PATH, model_config_path]
    for benchmark_path in benchmark_path_list:
        shutil.rmtree(benchmark_path, ignore_errors=True)
        os.makedirs(benchmark_path, exist_ok=True)

        print(f"successfully setup benchmark_path={benchmark_path}")


def build_model_json_config(models):
    for model in models:
        if model.startswith("/"):
            input_file = model
        else:
            input_file = f"{CWD}/benchmarks/models_config/{model}"
        gen_model_config_json.convert_yaml_to_json(input_file, MODEL_JSON_CONFIG_PATH)


def enable_launcher_with_logical_core(hw):
    if hw == "cpu":
        with open("./benchmarks/config.properties", "a") as f:
            f.write("cpu_launcher_enable=true\n")
            f.write("cpu_launcher_args=--use_logical_core\n")


def run_benchmark(bm_config):
    files = os.listdir(bm_config["model_config_path"])
    files.sort()
    for model_json_config in files:
        if model_json_config.endswith(".json"):
            # call benchmark-ab.py
            shutil.rmtree(TS_LOGS_PATH, ignore_errors=True)
            shutil.rmtree(BENCHMARK_TMP_PATH, ignore_errors=True)
            cmd = f'python ./benchmarks/benchmark-ab.py --tmp_dir /tmp --report_location /tmp --config_properties ./benchmarks/config.properties --config {bm_config["model_config_path"]}/{model_json_config}'
            execute(cmd, wait=True)

            # generate stats metrics from ab_report.csv
            bm_model = model_json_config[:-len(".json")]

            gen_metrics_json.gen_metric(
                f"{BENCHMARK_TMP_PATH}/ab_report.csv",
                f"{BENCHMARK_TMP_PATH}/logs/stats_metrics.json",
            )

            # load stats metrics to remote metrics storage
            if "metrics_cmd" in bm_config:
                execute(bm_config["metrics_cmd"], wait=True)

            # cp benchmark logs to local
            bm_model_log_path = f"{BENCHMARK_REPORT_PATH}/{bm_model}"
            os.makedirs(bm_model_log_path, exist_ok=True)
            csv_file = f"{BENCHMARK_TMP_PATH}/ab_report.csv"
            if os.path.exists(csv_file):
                shutil.move(csv_file, bm_model_log_path)
            cmd = f"tar -cvzf {bm_model_log_path}/benchmark.tar.gz {BENCHMARK_TMP_PATH}"
            execute(cmd, wait=True)

            cmd = f"tar -cvzf {bm_model_log_path}/logs.tar.gz {TS_LOGS_PATH}"
            execute(cmd, wait=True)
            print(f"finish benchmark {bm_model}")

    # generate final report
    gen_md_report.iterate_subdir(
        BENCHMARK_REPORT_PATH,
        f"{BENCHMARK_REPORT_PATH}/report.md",
        bm_config["hardware"],
        bm_config["version"],
    )
    print("report.md is generated")

    # load logs to remote storage
    if "report_cmd" in bm_config:
        execute(bm_config["report_cmd"], wait=True)


def clean_up_benchmark_env(bm_config):
    shutil.rmtree(BENCHMARK_TMP_PATH, ignore_errors=True)
    shutil.rmtree(MODEL_JSON_CONFIG_PATH, ignore_errors=True)
    shutil.rmtree(MODEL_STORE, ignore_errors=True)
    shutil.rmtree(WF_STORE, ignore_errors=True)


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(f"execute: {command}")
    cmd = Popen(
        command,
        shell=shell,
        close_fds=True,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
    )
    if wait:
        cmd.wait()
    return cmd


def get_torchserve_version():
    # fetch the torchserve version from version.txt file
    with open(os.path.join(CWD, "ts", "version.txt"), "r") as file:
        version = file.readline().rstrip()
    return version


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="benchmark config yaml file path",
    )
    parser.add_argument(
        "--skip",
        action="store",
        help="true: skip torchserve installation. default: true",
    )
    parser.add_argument(
        "--skip_upload",
        help="true: skip uploading commands . default: false",
    )

    arguments = parser.parse_args()
    skip_ts_config = arguments.skip is None or arguments.skip.lower() != "false"
    skip_upload = (
        arguments.skip_upload is not None
        and arguments.skip_upload.lower() == "true"
    )
    bm_config = load_benchmark_config(arguments.input, skip_ts_config, skip_upload)
    benchmark_env_setup(bm_config, skip_ts_config)
    run_benchmark(bm_config)
    clean_up_benchmark_env(bm_config)
    print("benchmark_serving.sh finished successfully.")


if __name__ == "__main__":
    main()
