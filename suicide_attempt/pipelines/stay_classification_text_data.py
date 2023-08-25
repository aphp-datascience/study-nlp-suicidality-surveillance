import os
from typing import Optional

import numpy as np
import pandas as pd
import typer

from suicide_attempt.functions import utils
from suicide_attempt.functions.data_wrangling import (
    add_hospital_label,
    cast_sex_cd_to_cat,
    compute_age_partition,
    compute_duration_and_year,
    reduce_rule_based_attributes,
    tag_recurrent_visit,
)
from suicide_attempt.functions.ehr_deployement import weight_stay


def stay_classification_text_data(
    conf_name: str, text_classification_method: Optional[str] = None
) -> pd.DataFrame:
    """
    This pipeline aggregates from entity level to stay level. It will provide information at stay aggregation level, particularly the column `any_true_instance`.

    Also, some extra columns are added to the returned df, for example:
    ['date', 'sex_cd', 'age', 'age_cat', 'hospital_label'].

    Parameters
    ----------
    conf_name:  str, configuration file name. The `delta_min_visits` (string with the accepted format of pd.to_timedelta)
        will be imported. If you want to avoid this censoring, so the `delta_min_visits` should be None.

    text_classification_method: str. {'text_rule_based','text_ml'}. If 'text_ml', so the following file should exist:
        "~/cse_210013/data/{conf_name}/result_ent_classification_ml_{conf_name}"

    Returns
    -------
    stays: pd.DataFrame. A df with Suicide Attempt stays identified by NLP after tagging false positives
        due to term modifiers  and to censoring recurrent stays.

        The df has the following columns:
        ['encounter_num', 'patient_num', 'visit_start_date', 'visit_end_date',
        'note_ids', 'std_lexical_variants', 'n_std_lexical_variants',
        'any_true_instance', 'concept_cds', 'date', 'sex_cd','count_true_instance',
        'age', 'age_cat', 'hospital_label','n_docs_per_visit', 'mode_entree','mode_sortie',
        'has_history','sex_cd','type_visite',]
    """  # noqa: E501

    # Read parameters / conf
    parameters = utils.get_conf(conf_name)

    # Read results of preceding pipeline
    file_path = os.path.expanduser(
        f"~/cse_210013/data/{conf_name}/result_ent_classification_rule_based_{conf_name}"  # noqa: E501
    )

    df = pd.read_pickle(file_path)

    # Retrieve true instances of the lexical variant
    df = reduce_rule_based_attributes(
        df, method=parameters["text_classification_method"]
    )
    df["history_patient"] = np.logical_and(
        np.logical_not(np.logical_or.reduce((df.negated, df.family, df.hypothesis))),
        df.history,
    )

    # Classification method: rulebased or ML (camembert)
    if text_classification_method is None:
        text_classification_method = parameters["text_classification_method"]

    if text_classification_method == "text_ml":

        results_bert = pd.read_pickle(
            os.path.expanduser(
                f"~/cse_210013/data/{conf_name}/result_ent_classification_ml_{conf_name}"  # noqa: E501
            )
        )
        results_bert["bert_token_prediction"] = results_bert[
            "bert_token_prediction"
        ].astype(bool)

        df = df.merge(
            results_bert[["bert_token_prediction"]],
            left_index=True,
            right_index=True,
            validate="one_to_one",
        )

        df["is_true_instance"] = df["bert_token_prediction"].astype(bool)
    else:
        df["is_true_instance"] = df["rule_based_prediction"]

    # Aggregate by visit
    stays = (
        df.groupby("encounter_num")
        .agg(
            patient_num=("patient_num", "first"),
            visit_start_date=("visit_start_date", "first"),
            visit_end_date=("visit_end_date", "first"),
            hospital=("hospital", "first"),
            note_ids=("note_id", "unique"),
            std_lexical_variants=("std_lexical_variant", "unique"),
            n_std_lexical_variants=("std_lexical_variant", len),
            any_true_instance=("is_true_instance", "any"),
            count_true_instance=("is_true_instance", "sum"),
            concept_cds=("concept_cd", "unique"),
            n_docs_per_visit=("n_docs_per_visit", "first"),
            mode_sortie=("mode_sortie", "first"),
            mode_entree=("mode_entree", "first"),
            has_history=("history_patient", "any"),
            birth_date=("birth_date", "first"),
            sex_cd=("sex_cd", "first"),
            type_visite=("type_visite", "first"),
        )
        .reset_index()
    )

    # Collect all positive mentions by visit
    df = df.query("is_true_instance")
    positive_mentions = df.groupby("encounter_num", as_index=False).agg(
        std_positive_lexical_variants=("std_lexical_variant", list),
    )
    stays = stays.merge(positive_mentions, how="left", on="encounter_num")

    # Compute age
    stays = compute_age_partition(stays)

    # Keep only the month
    stays["date"] = stays["visit_start_date"].astype("datetime64[M]")

    # Cast sex_cd to category
    stays = cast_sex_cd_to_cat(stays)

    # Compute duration of stay and year
    stays = compute_duration_and_year(stays)

    # Add hospital label
    stays = add_hospital_label(stays)

    # Add boolean columns: `nlp_positive` and `positive_visit`
    # For text data
    n_true_instances = parameters["threshold_positive_instances"]

    stays["nlp_positive"] = np.where(
        (stays["count_true_instance"] >= n_true_instances),
        True,
        False,
    )

    # Add a global tag of the visit
    stays["positive_visit"] = stays["nlp_positive"]

    # Tag recurrent stays
    if parameters["delta_min_visits"] is not None:
        stays = tag_recurrent_visit(stays, delta=parameters["delta_min_visits"])
    else:
        stays["recurrent_visit"] = False

    # Add Tag, if visit ends with death
    stays["death"] = np.where((stays["mode_sortie"] == "6-DC"), True, False)

    # Add column `stay_contribution`
    # (rescaling for the sensitivity analysis)
    stays = weight_stay(stays)

    return stays


def main(conf_name: str = typer.Argument(..., help="name of conf file")):
    """
    If __name__=='__main__': the entire pipeline will be run with the data
    of the specified configuration file.
    The results will be saved at
    'cse_210013/data/{CONFIG_FILE_NAME}/stay_classification_text_data_{CONFIG_FILE_NAME}'
    """

    # Run pipeline stay_classification_text_data
    df = stay_classification_text_data(conf_name=conf_name)

    # Save results
    path = os.path.expanduser(
        f"~/cse_210013/data/{conf_name}/stay_classification_text_data_{conf_name}"
    )
    df.to_pickle(path)

    print(
        "## Pipeline `stay_classification_text_data` succeeded ##\nResults saved at:",
        path,
    )


if __name__ == "__main__":
    typer.run(main)
