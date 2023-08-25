import json
import os
import subprocess

import click


def create_env(name, py, no_spark):

    user = os.popen("echo $USER").read().replace("\n", "")

    if not no_spark and str(py) >= "3.8.0":
        new_py = input(
            """You are trying to install a Spark-compatible environment with a python version > 3.7.1, which isn't compatible with the current Spark version of the cluster. Do you want to use python 3.7.1 ? [[y]/n]"""
        )
        if new_py is None or new_py == "y":
            py = "3.7.10"
        else:
            raise ValueError("Abort: Incompatible python version for Spark")

    s = f"""conda create  --p /export/home/{user}/.user_conda/miniconda/envs/{name}  python={py}
/export/home/{user}/.user_conda/miniconda/envs/{name}/bin/pip install ipykernel
/export/home/{user}/.user_conda/miniconda/envs/{name}/bin/python -m ipykernel install --user --name={name}"""

    if not no_spark:
        s += f"\n/export/home/{user}/.user_conda/miniconda/envs/{name}/bin/pip install pyspark==2.4.3"

    for command in s.split("\n"):
        print(f"COMMAND : {command}")
        subprocess.check_call(command, shell=True)

    return


def modify_kernel(name, no_spark):
    """
    Modify 2 things:
    - The python path
    - Some pyspark arguments
    """

    user = os.popen("echo $USER").read().replace("\n", "")

    default_kernel_path = f"/export/home/{user}/.local/share/jupyter/kernels/k8s-pyspark-client-2.4.3/kernel.json"
    kernel_path = f"/export/home/{user}/.local/share/jupyter/kernels/{name}/kernel.json"
    python_path = f"/export/home/{user}/.user_conda/miniconda/envs/{name}/bin/python"

    with open(default_kernel_path) as default_kernel:
        env_dict = json.load(default_kernel)["env"]

    for k in env_dict.keys():
        env_dict[k] = env_dict[k].replace(f"conda_{user}", name)
        env_dict[k] = env_dict[k].replace(
            ".user_conda/envs", ".user_conda/miniconda/envs"
        )

    CLASSPATH = os.popen("echo `$HADOOP_HOME/bin/hdfs classpath --glob`").read()
    env_dict["ARROW_LIBHDFS_DIR"] = "/usr/local/hadoop/usr/lib/"
    env_dict["HADOOP_HOME"] = "/usr/local/hadoop"
    env_dict["CLASSPATH"] = CLASSPATH

    with open(kernel_path) as file:
        kernel_dict = json.load(file)

    if not no_spark:
        kernel_dict["env"] = env_dict

    argv = kernel_dict["argv"]
    argv[0] = python_path
    kernel_dict["argv"] = argv

    with open(kernel_path, "w") as file:
        json.dump(kernel_dict, file)


@click.command()
@click.option(
    "--name",
    "-n",
    help="Name of the conda environment and jupyter kernel",
    prompt="Name of your environment:",
)
@click.option(
    "--py", "-p", default="3.7.10", help="Which python version to use (default 3.7.10)"
)
@click.option(
    "--no_spark",
    is_flag=True,
    default=False,
    help="If set, no Spark configuration will be added (lightweight install)",
)
def run(name, py, no_spark):

    print("Creating env...")
    create_env(name, py, no_spark)
    print("Modifying Kernel...")
    modify_kernel(name, no_spark)
    print("Done !")


if __name__ == "__main__":
    run()
