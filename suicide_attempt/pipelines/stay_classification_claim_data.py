import os

import numpy as np
import pandas as pd
import typer
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from suicide_attempt.functions import utils
from suicide_attempt.functions.data_wrangling import (
    add_hospital_label,
    cast_sex_cd_to_cat,
    compute_age_partition,
    compute_duration_and_year,
    tag_recurrent_visit,
)
from suicide_attempt.functions.retrieve_data import (
    get_patient_info,
    retrieve_stays,
    retrieve_stays_by_icd,
)


def _rule_retrieve_stays_by_icd(
    rule: str, schema: str, icd10_type: str
) -> SparkDataFrame:
    """
    Function to return stays that fulfill different definitions of suicide attempt by ICD10.

    Rules
    -----
    - "X60-X84": Will return a the stays that have at least one ICD code that belongs to the range X60 to X84.
    - "Haguenoer2008": Will return a the stays that follow the definiton of
        "Haguenoer, Ken, Agnès Caille, Marc Fillatre, Anne Isabelle Lecuyer, et Emmanuel Rusch. « Tentatives de Suicide », 2008, 4."
        This rule requires at least one Main Diagnostic (DP) belonging to S00 to T98,
        and at least one Associated Diagnostic (DAS) that belongs to the range X60 to X84.

    Parameters
    ----------
    rule: str,
        name of the rule to use. Possible values : {"X60-X84", "Haguenoer2008", "custom1"}
    schema: str,
        name of the schema to use.

    Returns
    -------
    pyspark.sql.DataFrame
    """  # noqa: E501
    if rule == "X60-X84":
        icd10 = retrieve_stays_by_icd(
            bloc_letter="X",
            bloc_number_min=60,
            bloc_number_max=84,
            schema=schema,
            icd10_type=icd10_type,
        )
        return icd10

    if rule == "Haguenoer2008":
        # Codes X
        icd10_X = retrieve_stays_by_icd(
            bloc_letter="X",
            bloc_number_min=60,
            bloc_number_max=84,
            schema=schema,
            icd10_type=icd10_type,
        )

        icd10_X = icd10_X.where(F.col("tval_char") == "DAS")

        # Codes S
        icd10_S = retrieve_stays_by_icd(
            bloc_letter="S",
            bloc_number_min=0,
            bloc_number_max=99,
            schema=schema,
            icd10_type=icd10_type,
        )

        icd10_S = icd10_S.where(F.col("tval_char") == "DP")

        # Codes T
        icd10_T = retrieve_stays_by_icd(
            bloc_letter="T",
            bloc_number_min=0,
            bloc_number_max=98,
            schema=schema,
            icd10_type=icd10_type,
        )

        icd10_T = icd10_T.where(F.col("tval_char") == "DP")

        # Union S and T
        icd10_ST = icd10_S.union(icd10_T)

        # Keep only the encounter_num column
        icd10_ST = icd10_ST.select("encounter_num")
        icd10_X = icd10_X.select("encounter_num")

        # Inner join to keep a df that fulfill both condition
        icd10 = icd10_X.join(icd10_ST, on=["encounter_num"], how="inner")

        return icd10


def stay_classification_claim_data(conf_name: str) -> pd.DataFrame:
    """
    Pipeline to identify stays by ICD10 codes.
    It will save the results at
    "~/cse_210013/data/{CONFIG_FILE_NAME}/stay_classification_claim_data_{CONFIG_FILE_NAME}"
    """
    # Read parameters
    parameters = utils.get_conf(conf_name)

    # Retrieve stays that have the subset of ICD10 diagnostics of the parameters
    icd10 = _rule_retrieve_stays_by_icd(
        rule=parameters["rule_icd10"],
        schema=parameters["schema"],
        icd10_type=parameters["icd10_type"],
    )

    # Select hospitals
    list_hospitals = parameters["hospitals_train"] + parameters["hospitals_test"]

    # Retrieve the stays to consider
    stays = retrieve_stays(
        list_admission_mode=None,
        list_type_of_visit=parameters["type_of_visit"],
        date_from=parameters["date_from"],
        list_hospitals=list_hospitals,
        schema=parameters["schema"],
        encounter_subset=parameters["encounter_subset"],
    )

    # Select columns
    stays = stays.withColumnRenamed("start_date", "visit_start_date")
    stays = stays.withColumnRenamed("end_date", "visit_end_date")
    stays = stays.select(
        [
            "patient_num",
            "encounter_num",
            "visit_start_date",
            "visit_end_date",
            "hospital",
            "mode_sortie",
            "mode_entree",
            "type_visite",
        ]
    )

    # Drop duplicate stays in icd10 df
    icd10_dedup = icd10.groupby("encounter_num").agg(
        F.collect_set("number_code_cim10").alias("number_codes_icd10")
    )

    # Join with the perimeter of stays
    stays_icd10 = stays.join(icd10_dedup, on=["encounter_num"], how="inner")

    # Add patient info
    stays_icd10 = get_patient_info(stays_icd10, parameters["schema"])

    # Add a debug option
    try:
        if parameters["debug"]:
            stays_icd10 = stays_icd10.limit(300)
    except:  # noqa: E722
        pass

    stays_icd10 = stays_icd10.toPandas()

    # Compute age
    stays_icd10 = compute_age_partition(stays_icd10)

    # Cast sex_cd to category
    stays_icd10 = cast_sex_cd_to_cat(stays_icd10)

    # Keep only the month
    stays_icd10["date"] = stays_icd10["visit_start_date"].astype("datetime64[M]")

    # Compute duration of stay and year
    stays_icd10 = compute_duration_and_year(stays_icd10)

    # Add hospital label
    stays_icd10 = add_hospital_label(stays_icd10)

    # Add tag positive visit
    stays_icd10["icd10_positive"] = True
    stays_icd10["positive_visit"] = True

    # Tag recurrent stays
    if parameters["delta_min_visits"] is not None:
        stays_icd10 = tag_recurrent_visit(
            stays_icd10, delta=parameters["delta_min_visits"]
        )
    else:
        stays_icd10["recurrent_visit"] = False

    # Add Tag, if visit ends with death
    stays_icd10["death"] = np.where((stays_icd10["mode_sortie"] == "6-DC"), True, False)

    return stays_icd10


def main(conf_name: str = typer.Argument(..., help="name of conf file")):
    """
    Pipeline to identify stays by ICD10 codes.
    It will save the results at
    "~/cse_210013/data/{conf_name}/stay_classification_claim_data_{conf_name}"
    """

    # Execute pipeline
    df = stay_classification_claim_data(conf_name=conf_name)

    # Export Results
    path = os.path.expanduser(
        f"~/cse_210013/data/{conf_name}/stay_classification_claim_data_{conf_name}"
    )
    df.to_pickle(path)

    print(
        "## Pipeline Stay classification claim data succeeded ##\nResults saved at:",
        path,
    )


if __name__ == "__main__":
    typer.run(main)
