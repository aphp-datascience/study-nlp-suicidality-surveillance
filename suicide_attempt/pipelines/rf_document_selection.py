import os

import typer
from pyspark.sql import functions as F

from suicide_attempt.functions.constants import regex_rf
from suicide_attempt.functions.retrieve_data import (
    get_patient_info,
    retrieve_table,
    spark,
)
from suicide_attempt.functions.stats_utils import get_sa_data
from suicide_attempt.functions.text_utils import spark_filter_docs_by_regex
from suicide_attempt.functions.utils import get_conf


def rf_document_selection(conf_name):
    """
    Pipeline to retrieve documents that mention a lexical variant of patient RF.
    We look only to stays classified as being caused by suicide attempt.

    Parameters
    ----------
    conf_name: str,
        Name of the configuration file.
        The configuration file should be placed at the folder
         `conf` and be in json format.

    Returns
    -------
    df_patient_rf: spark DataFrame
    """
    # Import parameters
    parameters = get_conf(conf_name)

    # Use the same docs retrieved in pipe 'stay_document_selection'
    documents = spark.read.parquet(
        f"cse_210013/pipeline_results/stay_document_selection_{conf_name}.parquet"
    )
    cols = ["note_id", "encounter_num", "patient_num", "concept_cd", "note_text"]
    documents = documents.select(cols)

    if parameters["debug"]:
        documents = documents.orderBy(F.rand())
        documents = documents.limit(2000)

    # keep only docs related to positive SA stays
    visits = get_sa_data(
        conf_name,
        source=parameters["text_classification_method"],
        keep_only_positive=True,
    )
    visits_nlp = visits.loc[visits.nlp_positive].encounter_num.to_list()
    documents = documents.where(F.col("encounter_num").isin(visits_nlp))
    documents = documents.drop_duplicates(subset=["note_id"])  # just to be sure

    # Filter documents by regex of RF
    df_patient_rf = spark_filter_docs_by_regex(
        documents, regex_pattern_dictionary=regex_rf
    )

    # Add visit_start_date
    stays = retrieve_table(schema=parameters["schema"], table_name="visits")
    stays = stays.select(["encounter_num", "start_date"])
    stays = stays.withColumnRenamed("start_date", "visit_start_date")
    df_patient_rf = df_patient_rf.join(stays, on="encounter_num", how="left")

    # Add birth_date
    df_patient_rf = get_patient_info(df_patient_rf, schema=parameters["schema"])

    return df_patient_rf


def main(conf_name: str = typer.Argument(..., help="name of conf file")):
    """
    Saves a spark DataFrame at the HDFS (one line per document)

    Results are saved at:
    hdfs://bbsedsi/user/{USER}/cse_210013/pipeline_results/rf_document_selection_{CONFIG_FILE_NAME}.parquet
    """
    # Execute pipeline
    df_patient_rf = rf_document_selection(conf_name=conf_name)

    # Save at the hdfs
    USER = os.environ["USER"]
    file_name = f"hdfs://bbsedsi/user/{USER}/cse_210013/pipeline_results/rf_document_selection_{conf_name}.parquet"  # noqa: E501
    df_patient_rf.write.mode("overwrite").parquet(file_name)
    print("Pipeline results saved at: ", file_name)


if __name__ == "__main__":

    typer.run(main)
