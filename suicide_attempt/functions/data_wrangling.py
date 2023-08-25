import os
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pandas import Int64Dtype
from pandas.arrays import IntervalArray
from pandas.core.frame import DataFrame

from suicide_attempt.functions.constants import age_partition, dict_code_UFR, regex_rf


def check_gb_cols(gb_cols=None):
    if gb_cols is not None:
        if ("sex_cd" in gb_cols) & ("age_cat" in gb_cols):
            assert gb_cols.index("sex_cd") < gb_cols.index(
                "age_cat"
            ), "`sex_cd` should be before than `age_cat` in gb_cols"


def _cast_as_category(
    df: pd.DataFrame, col: str, new_order: list, cats: list = None
) -> pd.DataFrame:
    """Cast a column as category and reorder the category levels with new_order.
    If cats is not None, so the order is: cats + new_order

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    new_order : list
    cats : list, optional
        ,by default None

    Returns
    -------
    pd.DataFrame
    """
    df[col] = df[col].astype("category")
    df[col] = df[col].cat.reorder_categories(
        cats + new_order if cats is not None else new_order
    )

    return df


def sort_df_by_sex_age(df: pd.DataFrame) -> pd.DataFrame:
    # Interval array of age partition
    from pandas.arrays import IntervalArray

    age_order = []
    ia = IntervalArray.from_breaks(age_partition)
    for arr in ia:
        age_cat = f"{round(arr.left)+1}-{round(arr.right)}"
        age_order.append(age_cat)
    age_order.append("all")

    df = _cast_as_category(df, col="age", new_order=age_order)
    df = _cast_as_category(df, col="sex", new_order=["M", "W", "all"])
    df.sort_values(["sex", "age"], inplace=True)

    return df


def _stats_lv(df):
    """
    Function to print statistics about the number of lexical variants per note,
    number of notes, visits and patients.

    Parameters
    ----------
    df: pd.DataFrame with the columns: note_id, std_lexical_variant, encounter_num,
    patient_num
    """
    m = df.groupby("note_id")["std_lexical_variant"].count().mean()
    print("Number of lexical variants per note:", m)
    print("Number of notes:", df.note_id.nunique())
    print("Number of visits:", df.encounter_num.nunique())
    print("Number of patients:", df.patient_num.nunique())


def reduce_rule_based_attributes(df, method, verbose=False):
    """
    Function to add a global tag to instances that are tagged as neither negated, family
    context, history, hypothesis nor reported speech (default behavior).
    Alternative tagging rules are implemented for risk factors.

    Parameters
    ----------
    df: pd.DataFrame with the boolean columns: negated, family, history, hypothesis.

    Returns
    -------
    df: the same dataframe with a boolean column `rule_based_prediction`.
    """

    if method == "rf":
        df["rule_based_prediction"] = np.logical_not(
            np.logical_or.reduce(
                (
                    df.negated,
                    df.hypothesis,
                )
            )
        )

    else:
        df["rule_based_prediction"] = np.logical_not(
            np.logical_or.reduce(
                (df.negated, df.family, df.history, df.hypothesis, df.rspeech)
            )
        )

    df["rule_based_prediction"] = df["rule_based_prediction"].astype(bool)

    df_keep = df.loc[df.rule_based_prediction]

    if verbose:
        print("### Notes retrieved by regex ###")
        _ = _stats_lv(df)

        print("\n### After tag true instances ###")
        _ = _stats_lv(df_keep)

    return df


def import_patient_rf_data(
    conf_name,
    subcats=None,
    text_modifers=["negated", "history", "hypothesis", "family", "rspeech"],
) -> DataFrame:
    """
    Import data related to Patient Risk Factors.
    Only data that is False in all of `text_modifers` is returned.

    Parameters
    ----------
    conf_name: str,
        Name of the configuration file
    subcats: None or list, default=None,
        subcategories of patient_context types.
    text_modifiers: list, default=['negated','history','hypothesis','family',
        'rspeech'],
        categories of text modifiers to consider. Selected categories are use to exclude
        instances.

    Returns
    -------
    df_patient_context: pd.DataFrame

    """
    path = os.path.expanduser(
        f"~/cse_210013/data/{conf_name}/result_classification_rf_{conf_name}"
    )
    df_patient_context = pd.read_pickle(path)

    if subcats is not None:
        df_patient_context = df_patient_context.loc[
            df_patient_context.std_lexical_variant.isin(subcats)
        ].copy()

    if text_modifers is not None:
        df_patient_context = df_patient_context.loc[
            np.logical_not(
                np.logical_or.reduce(df_patient_context[text_modifers], axis=1)
            )
        ]

    return df_patient_context


def _filter_auto_join(df, delta):
    """
    Auxiliray function to filter results after an auto-join
    """
    df = df.loc[
        (df.encounter_num_1 != df.encounter_num_2)
        & (df.delta <= delta)
        & (df.visit_start_date_1 <= df.visit_start_date_2)
    ]
    return df


def _compute_connected(G) -> pd.Series:
    """
    Auxiliary function to compute connected components
    """

    # Get connected components
    list_components = [i for i in nx.connected_components(G)]

    s = pd.Series(list_components)

    s = s.explode()

    s = pd.DataFrame(s)

    s_g = s.groupby(s.index).first()

    s = s.merge(s_g, how="left", left_index=True, right_index=True)

    s.columns = ["encounter_num", "stay_id"]
    return s


def tag_recurrent_visit(df, delta) -> pd.DataFrame:

    """
    Function to tag a visit with the value `recurrent_visit`=True if the time interval
    between the end of the previous visit and the start of the current one is less
    equal to `delta`, and the second visit starts after the first one.
    It will consider chains of visits also.
    A filter is done to consider only visits with any true instance

    Parameters
    ----------
    df: pd.DataFrame with the columns `visit_start_date`, `visit_end_date`,
    `patient_num`, `encounter_num`.
    mode: str, one of {'nlp','icd10'}. Which source of classification consider
    to chain the visits.
    delta: str. A string with the accepted format of pd.to_timedelta.

    Returns
    -------
    df: same dataframe with the boolean column `recurrent_visit`.
    """
    df = df.copy()

    # Filter visits in each case
    visits = df.loc[
        df.positive_visit,
        ["patient_num", "encounter_num", "visit_start_date", "visit_end_date"],
    ]

    # Join all visits of a patient
    visits_join = visits.merge(visits, on="patient_num", suffixes=("_1", "_2"))

    # Compute delta between visits
    visits_join["delta"] = (
        visits_join["visit_start_date_2"] - visits_join["visit_end_date_1"]
    )

    # Filter the visits in the auto_join table
    visits_join = _filter_auto_join(visits_join, delta=delta)

    if len(visits_join) > 0:
        # Build Graph
        G = nx.from_pandas_edgelist(
            visits_join[["encounter_num_1", "encounter_num_2"]],
            "encounter_num_1",
            "encounter_num_2",
        )

        # Get connected components
        s = _compute_connected(G)

        # Add the date of the visit
        s = s.merge(df[["encounter_num", "visit_start_date"]], on="encounter_num")

        # Sort values by stay_id and date
        s.sort_values(["stay_id", "visit_start_date"], ascending=True, inplace=True)

        # Tag the first visit of the stay as recurrent=False
        s_g = s.groupby("stay_id").agg(encounter_num=("encounter_num", "first"))
        s_g["recurrent_visit"] = False

        # Add this info to the table of stays-visits
        s = s.merge(s_g, on="encounter_num", how="left")

        # Tag the other visits (not the first) as recurrent=True
        s.recurrent_visit.fillna(True, inplace=True)

        # Add this info to the main table
        s = s[["encounter_num", "recurrent_visit"]]
        df = df.merge(s, on="encounter_num", how="left")

        # Complete the column for the rest of the visits with recurrent=False
        df.recurrent_visit = df.recurrent_visit.fillna(False)
    else:
        df["recurrent_visit"] = False

    return df


def filter_df_by_source_sex_age(df, age=None, sex=None, source=None):
    """
    Function to filter a df by sex, age or source.
    """
    df_f = df
    if age is not None:
        df_f = df_f.loc[(df.age_cat == age)]
    if sex is not None:
        df_f = df_f.loc[(df.sex_cd == sex)]
    if source is not None:
        df_f = df_f.loc[(df.source == source)]
    return df_f


def process_date(dates):

    # drop non parsed dates and 'duration'
    dates.dropna(subset=["date_dt"], inplace=True)
    idx1 = dates.query("std_lexical_variant=='duration' ").index
    idx2 = dates.query(
        "std_lexical_variant=='relative' and visit_start_date.isna()"
    ).index
    idx = idx1.union(idx2)
    dates.drop(idx, inplace=True)

    # cast
    dates.year = dates.year.astype(pd.Int64Dtype())
    dates.month = dates.month.astype(pd.Int64Dtype())
    dates.day = dates.day.astype(pd.Int64Dtype())
    dates["date_dt"] = pd.to_datetime(dates.date_dt)

    # Cast date
    dates["date"] = dates.date_dt.dt.date

    # Tag when the date is the birth date;
    # it uses the column birth_date (administrative data)
    dates["is_birth_date"] = dates.birth_date == dates.date

    return dates


def link_entities_w_dates(
    df1: pd.DataFrame, df2: pd.DataFrame, exclude_birthdate: bool = False
) -> pd.DataFrame:
    """Link entities with dates. The match by note_id and sent_id is exact.
    Then keeps the nearest date in the sentence. The measure of distance is
    the absolute number of tokens from date.start and ent.start.

    It is possible to avoid making the link with the date of birth,
    setting exclude_birthdate to True.
    This will work with the column `is_birth_date`.

    - Both dataframes should have the following columns:
    [`note_id`, `sent_id`, `start`].
    - df2 should have the column `date` and `is_birth_date`
    - df1 should have the column `visit_start_date`

    Parameters
    ----------
    df1 : pd.DataFrame
        dataframe with entities
    df2 : pd.DataFrame
        dataframe with dates
    exclude_birthdate:  bool, default=False
        wether to exclude birthdates form linkage.

    Returns
    -------
    pd.DataFrame
        df1 with the extra columns `date`, `delta_to_start_visit` and `is_birth_date`
    """
    df1.sort_values(["start"], ascending=True, inplace=True)
    n0 = len(df1)

    if exclude_birthdate:
        df2 = df2.loc[np.logical_not(df2.is_birth_date)].copy()

    df2.sort_values(["start"], ascending=True, inplace=True)
    _df2 = df2[["note_id", "sent_id", "date", "start", "is_birth_date"]]

    ents_matched_dates = pd.merge_asof(
        df1,
        _df2,
        on="start",
        by=["note_id", "sent_id"],
        direction="nearest",
        suffixes=("", "_date"),
    )
    n1 = len(ents_matched_dates)

    assert n0 == n1

    ents_matched_dates.date = pd.to_datetime(ents_matched_dates.date)

    ents_matched_dates["delta_to_start_visit"] = (
        ents_matched_dates.visit_start_date - ents_matched_dates.date
    )

    return ents_matched_dates


def tag_history_w_date(
    ents: pd.DataFrame, threshold: str, exclude_birthdate: bool
) -> pd.DataFrame:
    """Method to tag the entity as history with the date extracted by NLP

    Parameters
    ----------
    ents : pd.DataFrame
        df with entities. Should have the columns:
        `delta_to_start_visit`,`is_birth_date`, `history`
    threshold : str
        timedelta between the date and the start of the visit
    exclude_birthdate : bool
        if date is birth date, so do not consider as an history.

    Returns
    -------
    pd.DataFrame
    """

    # Tag history if delta is greater than constant
    if exclude_birthdate:
        idx = ents.loc[
            (ents["delta_to_start_visit"] >= pd.to_timedelta(threshold))
            & (np.logical_not(ents.is_birth_date))
        ].index
    else:
        idx = ents.loc[ents["delta_to_start_visit"] >= pd.to_timedelta(threshold)].index
    ents.loc[idx, "history"] = True

    return ents


def split_process_and_link(
    df: pd.DataFrame, exclude_birthdate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split into entities and dates
    dates = df.query("span_type=='date'").dropna(axis=1, how="all").copy()
    ents = df.query("span_type!='date'").dropna(axis=1, how="all").copy()

    # Process dates
    dates = process_date(dates)

    # Link both
    ents = link_entities_w_dates(
        df1=ents, df2=dates, exclude_birthdate=exclude_birthdate
    )

    return ents, dates


def transform_rf_stays_to_long(df):
    # Keep only SA positive stays (by the ML algorithm)
    df2 = df.query("ml_positive")
    value_vars = list(regex_rf.keys()) + ["has_history"]
    df3 = df2.melt(
        id_vars=["note_id", "after_covid_outbreak", "encounter_num"],
        value_vars=value_vars,
        value_name="rf_prediction",
        var_name="rf_name",
    )

    df3["note_id_dedup"] = df3["note_id"] + "-" + df3["rf_name"]
    return df3


def process_for_rf_plot(df_sex, df_all_pop, window=3):

    cols_rf_ratio = [i + "_ratio" for i in regex_rf]
    cols_rf_ratio.append("has_history_ratio")
    new_names = [i + "_smooth" for i in cols_rf_ratio]

    df_all_pop["sex_cd"] = "all_pop"
    df_plot_rf = pd.concat([df_sex, df_all_pop]).reset_index(drop=True)
    df_plot_rf[new_names] = (
        df_plot_rf.groupby(
            [
                "sex_cd",
            ],
            as_index=False,
        )
        .rolling(window, center=True)[cols_rf_ratio]
        .mean()
        .reset_index(drop=True)
    )
    return df_plot_rf


def compute_age_partition(df: pd.DataFrame) -> pd.DataFrame:

    # Cast to datetime
    df["birth_date"] = pd.to_datetime(df.birth_date)
    df["visit_start_date"] = pd.to_datetime(df["visit_start_date"])

    # Compute age
    df["age"] = (df["visit_start_date"] - df["birth_date"]) / np.timedelta64(1, "Y")
    df["age"] = (np.floor(df.age)).astype(pd.Int64Dtype())
    df.drop(columns=["birth_date"], inplace=True)

    # Compute age partition
    ia = IntervalArray.from_breaks(age_partition)

    # Recompute age_cat. Cast the age to category.
    # ATTENTION: `age` should be an int to work as expected.
    assert df.age.dtype in [Int64Dtype(), int]
    df["age_cat"] = pd.cut(df.age, age_partition, include_lowest=False)

    # Rename values of category age and add Unkown
    for arr in ia:
        df["age_cat"] = df.age_cat.cat.rename_categories(
            {arr: f"{round(arr.left)+1}-{round(arr.right)}"}
        )

    df["age_cat"] = df.age_cat.cat.add_categories("Unknown")
    df["age_cat"].fillna("Unknown", inplace=True)
    return df


def cast_sex_cd_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    # Add unknown to column sex for missing values
    df.sex_cd.fillna("Unknown", inplace=True)
    df.sex_cd = df.sex_cd.astype("category")
    if "Unknown" not in df.sex_cd.cat.categories:
        df["sex_cd"] = df.sex_cd.cat.add_categories("Unknown")
    return df


def compute_duration_and_year(df: pd.DataFrame) -> pd.DataFrame:
    # Compute duration of stay and compute the year of the stay
    df["duration"] = df["visit_end_date"] - df["visit_start_date"]

    # Cast duration of the stay to seconds
    df["duration_seconds"] = df.duration.dt.total_seconds()

    # Compute year of the stay
    df["year"] = df.date.dt.year
    return df


def add_hospital_label(df: pd.DataFrame) -> pd.DataFrame:
    # Import labels of hospitls codes
    key_hospitals = pd.DataFrame(
        dict_code_UFR.items(), columns=["hospital", "hospital_label"]
    )

    # Add hospital label
    df = df.merge(key_hospitals, on="hospital", how="left")

    # Remove code hospital column
    df.drop(
        columns=[
            "hospital",
        ],
        inplace=True,
    )
    return df


def filter_out_unknown_age_sex(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    # Number of visits without sex & age information
    age_unkown = df.loc[(df.age < 0) | (df.age.isna())].index

    sex_unknown = (df.sex_cd == "Unknown") | (df.sex_cd == "U")

    if verbose:
        print(
            "Number of visits without age information:",
            len(age_unkown),
        )
        print(
            "Number of patients without age information:",
            df.loc[age_unkown].patient_num.nunique(),
        )
        print(
            "Number of visits without sex information:",
            len(df.loc[sex_unknown]),
        )
        print(
            "Number of patients without sex information:",
            df.loc[sex_unknown].patient_num.nunique(),
        )
    idx_out_of_range = df.loc[(df.age >= 0) & (df.age_cat == "Unknown")].index

    if verbose:
        print(
            "Number of visits not considered because they are out of the age interval: ",  # noqa: E501
            len(idx_out_of_range),
        )
        print(
            "Number of patients not considered because they are out of the age interval: ",  # noqa: E501
            df.loc[idx_out_of_range].patient_num.nunique(),
        )

    # Filter out visits without sex
    df = df.loc[np.logical_not(sex_unknown)]

    # Filter out visits without age
    df = df.loc[df.age_cat != "Unknown"]

    # Remove unused categories
    df.age_cat = df.age_cat.cat.remove_unused_categories()
    df.sex_cd = df.sex_cd.cat.remove_unused_categories()

    return df
