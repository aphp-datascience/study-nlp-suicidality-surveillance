"""
Module to intearact with the project EDS Temporal Variability that qualifies
the dynamic of EHR deployment within AP-HP hospitals
"""
import os
from typing import List

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

from suicide_attempt.functions.constants import dict_code_UFR, tables


class ProbeEHRDeployement:
    def from_spark(
        sql, only_type_visit: list, only_cat_docs: list, schema: str
    ) -> DataFrame:
        """
        Probe of the deployment of the main electronic health record (EHR) software.
        This function can be executed by "data curators" only and not by the
         "investigators", because of data access rights.

        Parameters
        ----------
        sql :
            pyspark sql class
        only_type_visit : list
            list of visit types. Ex: ["I",]
        only_cat_docs : list
            list of document categories
        schema : str
            database

        Returns
        -------
        DataFrame
        """

        # Define Schema and tables
        sql(f"use {schema}")

        table_visits = tables(schema, "visits")
        table_visit_details = tables(schema, "visit_detail")
        table_documents = tables(schema, "documents")

        # VD (Retrieve hospital of the visit)
        vd = sql(f"SELECT * FROM {table_visit_details}")
        vd = vd.withColumn(
            "care_site_id", F.substring(F.col("concept_cd"), pos=1, len=7)
        )
        vd = vd.withColumnRenamed("encounter_num", "visit_occurrence_id")
        vd = vd.groupby("visit_occurrence_id").agg(
            F.first("care_site_id", ignorenulls=True).alias("care_site_id")
        )

        # VO
        vo = sql(
            f"""SELECT * FROM {table_visits}
            WHERE
                sourcesystem_cd='ORBIS'
                AND i2b2_action = 2
            """
        )
        vo = vo.withColumnRenamed("start_date", "visit_start_datetime")
        vo = vo.withColumnRenamed("encounter_num", "visit_occurrence_id")
        vo = vo.withColumnRenamed("type_visite", "visit_source_value")

        # Notes & filter types
        on = sql(
            f"""SELECT encounter_num,
            instance_num as note_id,
            concept_cd  FROM {table_documents}
            WHERE
                observation_blob IS NOT NULL
                AND i2b2_action = 2
            """
        )

        # Reference ORBIS DOCUMENT
        # Mapping between cd_document and cd_n2 (chapter)
        refdoc = sql("select cd_document, cd_n2 from orbis_ref_document")
        chapter_doc = refdoc.groupby(["cd_document"]).agg(
            F.first("cd_n2", ignorenulls=True).alias("note_class_source_value")
        )
        chapter_doc = chapter_doc.withColumn(
            "concept_cd", F.concat(F.lit("CR:"), F.col("cd_document"))
        )

        on = on.join(chapter_doc, on="concept_cd", how="left")
        on = on.where(on.note_class_source_value.isin(only_cat_docs))
        on = on.withColumnRenamed("encounter_num", "visit_occurrence_id")

        # Probe
        note_hosp_i2b2 = (
            vo.join(vd, on="visit_occurrence_id", how="inner")
            .join(on, on=vo.visit_occurrence_id == on.visit_occurrence_id, how="left")
            .where(
                (~F.isnull(vo.visit_start_datetime))
                & (~F.isnull(vd.care_site_id))
                & (vo.visit_source_value.isin(only_type_visit))
            )
            .select(
                vo.visit_occurrence_id,
                vo.visit_source_value.alias("stay_type"),
                F.trunc(vo.visit_start_datetime, "MM").alias("date"),
                on.note_class_source_value,
                vd.care_site_id.alias("id_hopital"),
                on.note_id,
            )
            .groupby(["stay_type", "date", "id_hopital", "visit_occurrence_id"])
            .agg(F.first("note_id", ignorenulls=True).alias("has_doc"))
            .groupby(["stay_type", "date", "id_hopital"])
            .agg(
                F.count("visit_occurrence_id").alias("n_visit"),
                F.count("has_doc").alias("n_note"),
            )
            .withColumn("rho", (F.col("n_note") / F.col("n_visit")))
            .withColumn("probe_subtype", F.lit("all"))
            .withColumnRenamed("id_hopital", "care_site_id")
            #    .drop(F.col("n_note"))
            .toPandas()
        )

        key_hospitals = pd.DataFrame(
            dict_code_UFR.items(), columns=["care_site_id", "care_site_name"]
        )

        note_hosp_i2b2 = note_hosp_i2b2.merge(
            key_hospitals,
            on="care_site_id",
            how="left",
            validate="many_to_one",
        )

        note_hosp_i2b2["date"] = pd.to_datetime(note_hosp_i2b2["date"])
        note_hosp_i2b2["care_site_level"] = "hopital"

        return note_hosp_i2b2


def get_data_ehr_deployement(
    hospit_list: List[str] = None,
    date_max: str = None,
    date_min: str = None,
    name_file="ratio_doc_hospit_092022",
) -> pd.DataFrame:
    """
    Import data coming from the probe of the deployment of the main electronic health
    record (EHR) software

    Parameters
    ----------
    hospit_list : List[str], optional
        list of hospital trigrams to keep, by default None
    date_max : str, optional
        max date, not inclusive (>), by default None
    date_min : str, optional
        min date, inclusive (<=), by default None
    name_file : str, name of file with raw data
        default = ratio_doc_hospit_092022

    Returns
    -------
    pd.DataFrame
        with the columns "care_site_id", "date", "rho", "stay_weight","hospital_label"
    """

    # Import data: Hospitalisation stays with at least one discharge summary
    ratio_doc_stay = pd.read_pickle(
        os.path.expanduser(
            f"~/cse_210013/data/export/ehr_deployement/{name_file}.pickle"
        )
    )[["care_site_id", "date", "rho"]]

    ratio_doc_stay["stay_weight"] = 1 / ratio_doc_stay["rho"]

    # Import labels of hospitls codes
    key_hospitals = pd.DataFrame(
        dict_code_UFR.items(), columns=["care_site_id", "hospital_label"]
    )

    # Merge hospital labels
    ratio_doc_stay = ratio_doc_stay.merge(
        key_hospitals, on="care_site_id", how="left", validate="many_to_one"
    )

    if hospit_list is not None:
        ratio_doc_stay = ratio_doc_stay.loc[
            ratio_doc_stay.hospital_label.isin(hospit_list)
        ]

    if date_max is not None:
        ratio_doc_stay = ratio_doc_stay.loc[ratio_doc_stay.date < date_max]
    if date_min is not None:
        ratio_doc_stay = ratio_doc_stay.loc[ratio_doc_stay.date >= date_min]

    return ratio_doc_stay


def weight_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to weight each stay by the proportion of hospitalisation
    stays with at least one discharge summary (rescaling for the sensitivity analysis)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        same df with the column `stay_weight` added.
    """
    # Import data: Hospitalisation stays with at least one discharge summary
    ratio_doc_stay = get_data_ehr_deployement()
    ratio_doc_stay.drop(columns=["care_site_id", "rho"], inplace=True)

    # Merge with df
    df = df.merge(
        ratio_doc_stay,
        on=["date", "hospital_label"],
        how="left",
        validate="many_to_one",
    )

    return df
