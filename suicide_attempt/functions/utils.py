import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame

# from pyarrow import fs
from pyspark.sql import SparkSession


def get_dir_path(file):
    path_conf_file = os.path.dirname(os.path.realpath(file))
    return path_conf_file


def get_conf(conf: str):
    """
    Function to read the configuration files

    Parameters
    ----------
    conf: str,
        name of configuration file. The file should be at the folder ./conf

    Returns
    -------
    parameters: dict,
        Python dictionary of parameters.
    """
    # Read configuration
    path = os.path.abspath(
        os.path.join(
            os.path.expanduser("~/cse_210013/conf"),
            conf + ".json",
        )
    )

    with open(path) as file:
        parameters = json.load(file)

    return parameters


def initiate_spark():
    # Create Spark Session
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    sql = spark.sql
    return spark, sql


def build_path(file, relative_path):
    """
    Function to build an absolut path.

    Parameters
    ----------
    file: main file from where we are calling. It could be __file__
    relative_path: str,
        relative path from the main file to the desired output

    Returns
    -------
    path: absolute path
    """
    dir_path = get_dir_path(file)
    path = os.path.abspath(os.path.join(dir_path, relative_path))
    return path


def save_file(
    file,
    conf_name,
    name,
):

    path_dir = os.path.abspath(
        os.path.join(
            os.path.expanduser("~/cse_210013/data/export"),
            conf_name,
        )
    )

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    path = os.path.join(path_dir, conf_name + name)

    f = open(path, "w")
    f.write(file)
    f.close()

    print("file saved at ", path)


def read_parquet_from_hdfs(
    file_name, folder="cse_210013/pipeline_results/", columns=None
) -> DataFrame:
    """
    Function to read a parquet file from the HDFS

    Parameters
    ----------
    file_name: name of file (without the .parquet extension)
    folder: folder where file is located, default="cse_210013/pipeline_results/"

    Returns
    -------
    df: pd.DataFrame
    """
    t1 = time.time()
    user = os.environ["USER"]
    file = f"hdfs://bbsedsi/user/{user}/{folder}{file_name}.parquet"

    df = pq.read_table(file, columns=columns).to_pandas()
    t2 = time.time()
    print(f"Time to read file {(t2-t1):.3f} sec")
    print("Number of lines:", len(df))

    return df


def get_toy_data() -> pd.DataFrame:
    """Function to get some toy data

    Returns
    -------
    pd.DataFrame
    """

    toy_data = pd.DataFrame(
        {
            "visit_start_date": [
                "2020-01-10",
                "2020-01-11",
                "2020-01-10",
                "2020-01-12",
                "2020-03-10",
                "2020-09-10",
                "2021-11-10",
                "2020-10-01",
                "2020-10-09",
            ],
            "patient_num": [1, 1, 2, 3, 2, 1, 3, 4, 4],
            "has_history": [True, None, None, None, None, None, None, None, None],
            "positive_visit": [
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
            ],
        }
    )
    toy_data.visit_start_date = pd.to_datetime(toy_data.visit_start_date)
    toy_data.patient_num = toy_data.patient_num.astype(str)

    toy_data.sort_values(
        [
            "patient_num",
            "visit_start_date",
        ],
        ascending=True,
        inplace=True,
    )

    return toy_data


def extract_global_labels(results):
    labels = results.global_labels.iloc[0].keys()
    for label in labels:
        results.loc[:, label] = results.global_labels.str[label]

    return results
