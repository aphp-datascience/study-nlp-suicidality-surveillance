from typing import List, Union

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from suicide_attempt.functions.constants import dict_hopital, tables
from suicide_attempt.functions.utils import initiate_spark

spark, sql = initiate_spark()


def retrieve_table(schema: str, table_name: str):
    sql(f"use {schema}")
    table = tables(schema, table_name)

    df = sql(f"SELECT * FROM {table}")
    return df


def retrieve_stays(
    list_admission_mode=None,
    list_type_of_visit=None,
    date_from=None,
    date_to=None,
    list_hospitals=None,
    schema: str = None,
    encounter_subset=None,
):
    """
    Function to retrieve orbis visits from i2b2 schema.
    It is possible to specify some constraints.

    Parameters
    ----------
    list_admission_mode: list, default=None
        Admissions mode to keep. For example: ["2-URG"].
        If `None`, it will keep all possible values.
    list_type_of_visit: list, default=None
        Type of visit to keep. For example: ["I","U"].
        If `None`, it will keep all possible values.
    date_from: str, default=None
        Filter the visits where `start_date` >= date_from.
        If `None`, it will not apply this temporal filter.
    date_to: str, default=None
        Filter the visits where `start_date` < date_to (exclusive).
        If `None`, it will not apply this temporal filter.
    list_hospitals: list, default=None
        Hospitals of the visit. Limits to visits in these hospitals.
        The hospitals should be expressed with the trigram.
    schema: str, default='cse_210013_20210726'
        schema to use
    encounter_subet: list, default=None,
        list of encounter ids to restrict to only that group

    Returns
    -------
    visits: Spark DataFrame
        A spark dataframe resulting from the sql query and restrictions.
    """

    visits = retrieve_table(schema=schema, table_name="visits")
    visits = visits.where(col("sourcesystem_cd") == "ORBIS")

    if list_admission_mode is not None:
        visits = visits.where(col("mode_entree").isin(list_admission_mode))

    if list_type_of_visit is not None:
        visits = visits.where(col("type_visite").isin(list_type_of_visit))

    if date_from is not None:
        visits = visits.where(col("start_date") >= date_from)

    if date_to is not None:
        visits = visits.where(col("start_date") < date_to)

    if encounter_subset is not None:
        visits = visits.where(col("encounter_num").isin(encounter_subset))

    # Hospital
    visit_details = retrieve_table(schema=schema, table_name="visit_detail")
    visit_details = visit_details.withColumn(
        "hospital", F.substring(col("concept_cd"), pos=1, len=7)
    )
    if list_hospitals is not None:
        list_hospitals_transco = [dict_hopital.get(key) for key in list_hospitals]

        visit_details = visit_details.where(
            col("hospital").isin(list_hospitals_transco)
        )
    visit_details = visit_details.drop_duplicates(subset=["encounter_num"])
    visit_details = visit_details.select(["encounter_num", "hospital"])
    visits = visits.join(visit_details, how="inner", on=["encounter_num"])

    return visits


def retrieve_docs(
    patient_subset=None,
    concept_cd_subset=None,
    encounter_subset=None,
    exclude_cat=None,
    schema: str = None,
):
    """
    Function to retrieve text documents

    Parameters
    ----------
    patient_subset: list, default=None
        List of patients ids. If `None`, no filter will be applied.
    concept_cd: list, default=None
        Type of concept codes of documents. If `None`, no filter will be applied.
    encounter_subset: list, default=None,
        Subset of encounter numbers.
    exclude_cat: list, default=None,
        Type of concept codes of documents to exclude.
        If `None`, no filter will be applied.
    schema: str
        schema to use

    Returns
    -------
    documents: Spark DataFrame
        A spark dataframe resulting from the sql query and restrictions.
    """
    sql(f"use {schema}")
    table_documents = tables(schema, "documents")

    documents = sql(f"SELECT * FROM {table_documents}")
    documents = documents.withColumnRenamed("observation_blob", "note_text")
    documents = documents.withColumnRenamed("instance_num", "note_id")
    documents = documents.withColumnRenamed("start_date", "document_datetime")

    if patient_subset is not None:
        documents = documents.where(col("patient_num").isin(patient_subset))
    if concept_cd_subset is not None:
        documents = documents.where(col("concept_cd").isin(concept_cd_subset))
    if encounter_subset is not None:
        print("Number of encounter ids: ", len(encounter_subset))
        if type(encounter_subset) == list:
            documents = documents.where(col("encounter_num").isin(encounter_subset))
        if type(encounter_subset) == pd.DataFrame:
            spark_encounter_ids = spark.createDataFrame(encounter_subset)
            documents = documents.join(
                spark_encounter_ids, on=["encounter_num"], how="inner"
            )
    if exclude_cat is not None:
        documents = documents.where(~F.col("concept_cd").isin(exclude_cat))

    return documents


def get_patient_info(ids_patient: Union[List[str], DataFrame], schema: str):
    """
    Returns a sparkDataframe with patient information: birth_date and sex

    Parameters
    ----------
    ids_patients: list or pyspark.sql.dataframe.DataFrame.
        list of patient_num of the table i2b2_patient_dim. If a DataFrame is passed,
        it should have the column `patient_num`.

    Returns
    -------
    patients:
        spark.DataFrame with the columns patient_num, birth_date, sex_cd.
        If `ids_patient` is a DataFrame,
        so these columns will be added to the original ones.
    """
    # Import data
    sql(f"use {schema}")
    table_patients = tables(schema, "patients")

    patients = sql(
        f"""SELECT
                patient_num,
                birth_date AS birth_datetime,
                sex_cd
            FROM {table_patients}"""
    )

    # Filter by id
    if type(ids_patient) == list:
        patients = patients.where(F.col("patient_num").isin(ids_patient))
    if type(ids_patient) == DataFrame:
        patients = ids_patient.join(patients, on=["patient_num"], how="left")

    # Convert datetime col to date
    patients = patients.withColumn("birth_date", F.to_date(F.col("birth_datetime")))
    patients = patients.drop("birth_datetime")

    return patients


def retrieve_stays_by_icd(
    bloc_letter: str,
    bloc_number_min: int,
    bloc_number_max: int,
    schema: str,
    icd10_type: str,
):
    """
    Retrieve a sparkDataFrame with visits that fulfill the specified ICD10 diagnoses.

    Parameters
    ----------
    bloc_letter: str,
        the chapter of the ICD-10 code to search.
        For example, if we want the codes between X60 and X84,
        then bloc_letter == "X"
    bloc_number_min: int,
        the first two numerical digits of the code
    icd10_type: str,
        One of {'ORBIS','AREM'}


    Returns
    -------
    Returns a spark.DataFrame with the visits and ICD10 codes that
    fulfill the conditions.

    Visits could be duplicated if they have multiple codes.

    With the following columns:
    [encounter_num,patient_num,instance_num,tval_char,
    concept_cd,letter_code_cim10,number_code_cim10]

    """
    # Set schema
    sql(f"USE {schema}")
    table_icd10 = tables(schema, "icd10")

    # Select table
    icd10 = sql(
        f"""SELECT
                encounter_num,
                patient_num,
                instance_num,
                tval_char,
                concept_cd
            FROM {table_icd10}"""
    )

    # Keep only AREM or ORBIS source software
    icd10 = icd10.where(F.col("sourcesystem_cd") == icd10_type)

    # Extract codes from column
    icd10 = icd10.withColumn("letter_code_cim10", F.substring("concept_cd", 7, 1))
    icd10 = icd10.withColumn(
        "number_code_cim10", F.substring("concept_cd", 8, 2).cast("int")
    )

    # Filter by codes
    icd10 = icd10.where(F.col("letter_code_cim10") == bloc_letter)
    icd10 = icd10.where(F.col("number_code_cim10") >= bloc_number_min)
    icd10 = icd10.where(F.col("number_code_cim10") <= bloc_number_max)

    return icd10


def keep_one_doc_per_visit(documents, method):
    """Auxiliary function to keep the first or last document of each visit.

    Parameters
    ----------
    documents : [type]
        spark dataframe with at least the following columns :
        ['document_datetime','encounter_num']
    method : str, {"keep_last", "keep_first"} or None. If None, so no action is applied.

    Returns
    -------
    documents
    """
    print("Method keep_one_doc_of_visit :", method)

    if method is not None:

        if method == "keep_last":
            _window = Window.partitionBy("encounter_num").orderBy(
                [
                    F.when(F.col("concept_cd") != "CR:INCONNU", 1)
                    .when(F.col("concept_cd") == "CR:INCONNU", 2)
                    .asc(),
                    F.col("document_datetime").desc(),
                    F.col("note_id").desc(),
                ]
            )

        if method == "keep_first":
            _window = Window.partitionBy("encounter_num").orderBy(
                [
                    F.when(F.col("concept_cd") != "CR:INCONNU", 1)
                    .when(F.col("concept_cd") == "CR:INCONNU", 2)
                    .asc(),
                    F.col("document_datetime").asc(),
                    F.col("note_id").desc(),
                ]
            )

        documents = documents.withColumn("row_number", F.row_number().over(_window))
        documents = documents.filter(F.col("row_number") == 1).drop("row_number")

    return documents
