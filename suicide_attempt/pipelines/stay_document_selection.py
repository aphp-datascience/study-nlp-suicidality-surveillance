import json
import os

import typer
from pyspark.sql import Window
from pyspark.sql import functions as F

from suicide_attempt.functions.constants import regex_sa
from suicide_attempt.functions.retrieve_data import (
    get_patient_info,
    keep_one_doc_per_visit,
    retrieve_docs,
    retrieve_stays,
)
from suicide_attempt.functions.text_utils import spark_filter_docs_by_regex


def stay_document_selection(conf_name: str, save: bool = True, debug_size: int = 300):
    """
    Pipeline to retrieve documents identified with the regex and that are associated
     to visits that fulfill some conditions.

    The conditions of the visits are :
       - Admission mode
       - Type of visit
       - Date from
       - Hospital

    If `rule_select_docs` is one of {'keep_last' , 'keep_first'},
     it will preselect the first or last document of the stay (and just one)
     before looking for the regex match.

    Parameters
    ----------
    conf_name: str,
        Name of the configuration file.
         The configuration file should be placed at the folder `conf` and be in json format.
        The dictionary should contain the following keys:
        admission_mode, type_of_visit, date_from, hospitals_train/test,
         schema, regex_lowercase, rule_select_docs.
        All hospitals (train and test) will be used.
    save: bool, whether to save the file at
    'hdfs://bbsedsi/user/{username}/cse_210013/pipeline_results/stay_document_selection_{conf_name}.parquet' or not,
      if False it will return a spark Dataframe.
        Default: True
    debug_size: int, default=300. Maximum number of documents to save.
     This parameter is only used if the `debug` key (at the conf file) is set to True

    Returns
    -------
    documents: spark DataFrame,
        The df contains the columns:
        ['encounter_num', 'patient_num', 'note_text', 'concept_cd', 'note_id',
         'visit_start_date', 'visit_end_date', 'hospital','birth_date', 'sex_cd']

    """  # noqa: E501

    # Read configuration
    path_conf_file = os.path.abspath(
        os.path.join(
            os.path.expanduser("~/cse_210013/conf"),
            conf_name + ".json",
        )
    )
    print("Configuration file: ", path_conf_file)
    with open(path_conf_file) as file:
        parameters = json.load(file)

    list_hospitals = parameters["hospitals_train"] + parameters["hospitals_test"]

    # Retrieve visits
    visits = retrieve_stays(
        list_admission_mode=parameters["admission_mode"],
        list_type_of_visit=parameters["type_of_visit"],
        date_from=parameters["date_from"],
        list_hospitals=list_hospitals,
        schema=parameters["schema"],
        encounter_subset=parameters["encounter_subset"],
    )

    # Select columns
    visits = visits.withColumnRenamed("start_date", "visit_start_date")
    visits = visits.withColumnRenamed("end_date", "visit_end_date")
    visits = visits.select(
        [
            "encounter_num",
            "visit_start_date",
            "visit_end_date",
            "hospital",
            "mode_sortie",
            "mode_entree",
            "type_visite",
        ]
    )

    # Retrieve documents
    documents = retrieve_docs(
        schema=parameters["schema"],
        concept_cd_subset=parameters["only_cat_docs"],
    )

    # Select columns
    documents = documents.select(
        [
            "patient_num",
            "encounter_num",
            "note_text",
            "concept_cd",
            "note_id",
            "document_datetime",
        ]
    )

    # Keep just documents that belong to the previously selected visits
    documents = documents.join(visits, on=["encounter_num"], how="inner")

    # Count number of documents per visit (selected categories)
    partition = Window.partitionBy("encounter_num")
    documents = documents.withColumn(
        "n_docs_per_visit", F.count("note_id").over(partition)
    )

    # option: Select just one document by visit
    if "rule_select_docs" in parameters.keys():
        documents = keep_one_doc_per_visit(
            documents=documents, method=parameters["rule_select_docs"]
        )

    # Filter documents by regex
    documents = spark_filter_docs_by_regex(documents, regex_pattern_dictionary=regex_sa)

    # Add patient info
    documents = get_patient_info(documents, schema=parameters["schema"])

    if "debug" in parameters.keys():
        if parameters["debug"]:
            documents = documents.orderBy(F.rand(seed=42))
            documents = documents.limit(debug_size)

    if save:
        print("Count Documents:", documents.cache().count())

        # Save results
        username = os.environ.get("USER")
        path_hdfs = f"hdfs://bbsedsi/user/{username}/cse_210013/pipeline_results/stay_document_selection_{conf_name}.parquet"  # noqa: E501
        documents.write.mode("overwrite").parquet(path_hdfs)
        print("Pipeline results saved at: ", path_hdfs)
    else:
        return documents


if __name__ == "__main__":
    typer.run(stay_document_selection)
