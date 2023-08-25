import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score
from statsmodels.iolib.summary import Summary
from statsmodels.stats.proportion import proportion_confint

from suicide_attempt.functions.constants import (
    alpha,
    covid_date,
    hospital_type_dict,
    regex_rf,
    seconds_per_day,
    unknown_and_other_forms,
)
from suicide_attempt.functions.data_wrangling import (
    check_gb_cols,
    filter_out_unknown_age_sex,
    import_patient_rf_data,
    process_for_rf_plot,
    sort_df_by_sex_age,
)
from suicide_attempt.functions.utils import extract_global_labels, get_conf, save_file
from suicide_attempt.pipelines.stay_classification_text_data import (
    stay_classification_text_data,
)


def _row_table1(
    reg: Summary,
    df_sex_age: pd.DataFrame,
    sex: str = "all",
    age: str = "all",
    y_col: str = "n_sa",
) -> Dict[str, Any]:
    """
    Populate each row of table 2 of the article

    Parameters
    ----------
    reg : Summary
        summary of the regression corresponding to sex and age
    df_sex_age : pd.DataFrame
        df of stype df_plot_texts_source_sex_age.
        It should have at least the following columns:
        {`sex_cd`, `age_cat`, `date`,`n_sa`}
    sex : str, optional
        any of {'W','M','all'}, by default "all"
    age : str, optional
        an interval that matchs with the age_partition. Example: '8-17'
        By default "all"
    y_col: column to sum
        default='n_sa'
    Returns
    -------
    Dict[str, Any]
        row in the form of a dictionary
    """

    # Results of regression
    results_as_html = reg.tables[1].as_html()
    s = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # Number of stays
    df_sex_age = df_sex_age.copy()

    total_postcovid = df_sex_age.query("post_covid")[y_col].sum()
    total_precovid = df_sex_age.query("~post_covid")[y_col].sum()

    if sex != "all":
        df_sex_age = df_sex_age.query(f"sex_cd=='{sex}'")
    if age != "all":
        df_sex_age = df_sex_age.query(f"age_cat=='{age}'")

    post_covid_n_sa = round(df_sex_age.query("post_covid")[y_col].sum())
    pre_covid_n_sa = round(df_sex_age.query("~post_covid")[y_col].sum())

    # Return dictionary
    r = {
        "sex": sex,
        "age": age,
        "pre_covid_n_sa": f"{pre_covid_n_sa} ({(pre_covid_n_sa/total_precovid*100):.1f}%)",  # noqa: E501
        "post_covid_n_sa": f"{post_covid_n_sa} ({(post_covid_n_sa/total_postcovid*100):.1f}%)",  # noqa: E501
        "pre_covid_mean": f"{s.loc['const','coef']:.1f} ({s.loc['const','[0.025']:.1f} - {s.loc['const','0.975]']:.1f})",  # noqa: E501
        "pre_covid_trend": f"{s.loc['t','coef']:.1f} ({s.loc['t','[0.025']:.1f} - {s.loc['t','0.975]']:.1f})",  # noqa: E501
        "post_covid_trend": f"{s.loc['delta_t','coef']:.1f} ({s.loc['delta_t','[0.025']:.1f} - {s.loc['delta_t','0.975]']:.1f})",  # noqa: E501
    }
    return r


def get_table1(
    reg_summary_sex_age: Dict[str, Summary],
    reg_summary_sex: Dict[str, Summary],
    reg_summary_all_pop: Summary,
    df_sex_age: pd.DataFrame,
    y_col: str = "n_sa",
) -> pd.DataFrame:
    """Function to get table 2 of the article

    Parameters
    ----------
    reg_summary_sex_age : Dict[str, Summary]
        summary of the regression fitted in data grouped by sex and age
    reg_summary_sex : Dict[str, Summary]
        summary of the regression fitted in data grouped by sex
    reg_summary_all_pop : Summary
        summary of the regression fitted in data of all pop
    df_sex_age : pd.DataFrame
        df of stype df_plot_texts_source_sex_age.
        It should have at least the following columns:
        {`sex_cd`, `age_cat`, `date`,y_col}
    y_col: column to sum
        default='n_sa'

    Returns
    -------
    pd.DataFrame
        table 2 of the article
    """
    # Add a column covid_date
    df_sex_age["post_covid"] = df_sex_age.date >= covid_date

    # Initialize list for to save each row
    rows = []

    # Populate each row for sex & age
    for key in reg_summary_sex_age.keys():
        sex = key[:1]
        age = key[2:]
        row = _row_table1(
            sex=sex,
            age=age,
            reg=reg_summary_sex_age[key],
            df_sex_age=df_sex_age,
            y_col=y_col,
        )
        rows.append(row)

    # Populate each row for sex
    for sex in ["W", "M"]:
        row = _row_table1(
            sex=sex,
            age="all",
            reg=reg_summary_sex[sex],
            df_sex_age=df_sex_age,
            y_col=y_col,
        )
        rows.append(row)

    # Populate each row all sex & all age
    row = _row_table1(
        sex="all",
        age="all",
        reg=reg_summary_all_pop,
        df_sex_age=df_sex_age,
        y_col=y_col,
    )
    rows.append(row)

    # Build table 2
    table1 = pd.DataFrame(rows)
    table1 = sort_df_by_sex_age(table1)
    table1.reset_index(inplace=True, drop=True)

    # Drop column
    df_sex_age.drop("post_covid", axis=1, inplace=True)

    return table1


def _get_table3(
    df: pd.DataFrame,
    rf_dict: Dict[str, str] = {
        "sexual_violence": "Sexual violence",
        "has_history": "Suicide attempt history",
        "domestic_violence": "Domestic violence",
        "physical_violence": "Physical violence",
        "social_isolation": "Social isolation",
    },
    _columns: str = "after_covid",
    incidence_rate: str = "n_sa",
    alpha: float = alpha,
    group: str = None,
    verbose: bool = False,
) -> pd.DataFrame:

    df = df.copy()
    # Make boolean column
    df["after_covid"] = df.date >= covid_date

    rows = []
    for rf in rf_dict.keys():
        negative_rf = f"non_{rf}"

        # Negative RF as the difference of the N cases - True RF
        df[negative_rf] = df[incidence_rate] - df[rf]

        assert (df[incidence_rate] >= df[rf]).all()  # sanity check

        # Compute contingency table
        ct = pd.pivot_table(
            df,
            values=[
                negative_rf,
                rf,
            ],
            columns=[
                _columns,
            ],
            aggfunc=np.sum,
        )

        # Perform fisher test
        _, pvalue_fisher_test = fisher_exact(
            table=ct.to_numpy(), alternative="two-sided"
        )

        # Subtotals
        subtotals = ct.sum(axis=0)

        # Divide ct by subtotals
        normalized = ct / subtotals

        # Reorder ct to be in good format for statsmodel
        ct22 = ct.T[[rf, f"non_{rf}"]].sort_index(ascending=False)

        # make a statsmodel Table2x2
        t22 = sm.stats.Table2x2(ct22.to_numpy())

        # Get the confidence interval
        ci = t22.riskratio_confint(alpha=alpha)

        if verbose:
            # Print
            print("\n######\n", rf)
            print(t22.summary())

            print(ct22)

        r = {
            "risk_factor": rf_dict[rf],
            "pre_covid_prop": f"{(normalized.loc[rf, False]):.2f} ({ct.loc[rf, False]})",  # noqa: E501, E261
            "post_covid_prop": f"{(normalized.loc[rf, True]):.2f} ({ct.loc[rf, True]})",  # noqa: E501, E261
            "prevalence_ratio_ci": f"{(t22.riskratio):.1f} ({ci[0]:.2f} - {ci[1]:.2f})",  # noqa: E501, E261
            "p_value_fisher_test": f"{pvalue_fisher_test}",  # noqa: E501
        }

        rows.append(r)

    table3 = pd.DataFrame(rows)

    if group is not None:
        table3["Group"] = group

    return table3


def get_table3(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Wrapper function to get table 3 of the article

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        table 3 of the article
    """

    tb3_all = _get_table3(
        df,
        group="Overall population",
        **kwargs,
    )

    tb3_w = _get_table3(df.query("sex_cd=='W'"), group="W", **kwargs)
    tb3_m = _get_table3(df.query("sex_cd=='M'"), group="M", **kwargs)

    tb3 = pd.concat([tb3_all, tb3_w, tb3_m])
    tb3.Group = pd.Categorical(
        tb3.Group, categories=["Overall population", "M", "W"], ordered=True
    )
    tb3 = tb3.sort_values(["risk_factor", "Group"]).reset_index(drop=True)
    tb3 = tb3.set_index(["risk_factor", "Group"])
    return tb3


def _preprocess_data_to_fit_linear_model(
    df: pd.DataFrame,
    outbreak_date: str,
    x_col_names: List[str],
    date_lower_limit: str,
    date_upper_limit: str,
    y_col_name: str = "n_sa",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.core.indexes.datetimes.DatetimeIndex]:
    """Function that preprocess data to be fed into
    the Linear Model defined as Equation 1 in article.

    Parameters
    ----------
    df : pd.DataFrame
        It should have the columns `date` and `n_sa`.
        All aggregations should be done before and granularity
        should be the `date` column (one point by month).
    outbreak_date : str
        date since parameters alpha_2 and alpha_3 are activated
    date_lower_limit : str
        date of start of data availability
    date_upper_limit : str
        max date available (non inclusive)
    y_col_name : str, default="n_sa"
        Name of the column to predict.
    x_col_names: List[str],
        The columns to be used

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.core.indexes.datetimes.DatetimeIndex]
    """
    # Verify that there is one point per date
    assert df.date.nunique() == len(df), "Data should be aggregated by date"

    # cast dates
    df = df.copy()
    outbreak_date = pd.to_datetime(outbreak_date)
    df["date"] = pd.to_datetime(df.date)

    # Sort values by date
    df = df.sort_values(by="date", ascending=True, inplace=False)

    # Reindex, assign 0 for missing values
    df = df[["date", y_col_name]]
    idx = pd.date_range(
        start=date_lower_limit, end=date_upper_limit, freq="MS", closed="left"
    )
    original_index = pd.DatetimeIndex(df.date)
    df.index = original_index
    df.drop(columns="date", inplace=True)
    df = df.reindex(idx, fill_value=0)
    df["date"] = df.index

    # Compute the number of months since t0
    t0 = pd.to_datetime(date_lower_limit)
    nb_monts = ((df.date - t0) / np.timedelta64(1, "M")).round().astype(int)
    df["t"] = nb_monts

    # Compute a binary variable that has the value 1 after outbreak_date
    df["after_outbreak"] = (df.date >= outbreak_date).astype(int)

    # t_since_outbreak if t >= t_outbreak else 0
    idx_after = df.query("after_outbreak==1").index
    t_ob = df.loc[idx_after].t.min()
    df.loc[idx_after, "delta_t"] = df["t"] - t_ob
    df["delta_t"] = df["delta_t"].fillna(0)
    df["delta_t"] = df["delta_t"].astype(int)

    # Make indicators columns by month
    df["month_name"] = df.date.dt.month_name()
    df["dummy"] = 1
    tmp_df = df.pivot(columns="month_name", values="dummy").fillna(0).astype(int)
    df = df.drop(columns=["dummy"])
    df = pd.concat([df, tmp_df], axis=1)

    assert (
        df.columns.is_unique
    ), f"Column Names are not unique. Column names: {df.columns}"

    X = df[x_col_names]
    y = df[y_col_name]

    return X, y, original_index


def fit_and_predict_linear_model_for_group(
    df: pd.DataFrame,
    outbreak_date: str,
    date_lower_limit: str,
    date_upper_limit: str,
    y_col_name: str = "n_sa",
    x_col_names: Optional[List[str]] = None,
    return_coef: bool = True,
) -> Union[Tuple[pd.DataFrame, Dict[str, float], Dict[str, Summary]], pd.DataFrame,]:
    """Function that computes a Linear Model defined as Equation 1 in article.


    Parameters
    ----------
    df : pd.DataFrame
        It should have the columns `date` and `n_sa`.
        All aggregations should be done before and
        granularity should be the `date` column (one point by month).
    outbreak_date : str
        date since parameters alpha_2 and alpha_3 are activated
    date_lower_limit : str
        date of start of data availability
    date_upper_limit : str
        max date available (non inclusive)
    y_col_name : str, default="n_sa"
        Name of the column to predict.
    x_col_names: Optional[List[str]], default=None.
        If None, the following columns are used:
        ['t', 'after_outbreak', 'delta_t', 'January', 'February',
        'March', 'April', 'May', 'June', 'July', 'September',
        'October', 'November', 'December']
    return_coef: bool, default=True
        Either to return the coeficient or just fit, predict and return dataframe.

    Returns
    -------
    If return_coef is True:
        Tuple[pd.DataFrame, Dict[str:float],Dict[str, Summary]]

            - df with features X as columns, estimated, and errors.
            - Dictionnary of Coefficients
            - Summary (summary of fit)
    Else:
    pd.DataFrame
        - df with features X as columns, estimated, and errors.

    """

    # Make arrays
    if x_col_names is None:
        x_col_names = [
            "t",
            "delta_t",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

    X, y, original_index = _preprocess_data_to_fit_linear_model(
        df=df,
        outbreak_date=outbreak_date,
        y_col_name=y_col_name,
        x_col_names=x_col_names,
        date_lower_limit=date_lower_limit,
        date_upper_limit=date_upper_limit,
    )

    # Fit regression
    X = sm.add_constant(X)
    mod = sm.OLS(y, X, hasconst=None)
    reg = mod.fit()
    coef = reg.params

    # Predict using the model
    original_X = X.loc[original_index].values
    df[f"estimated_{y_col_name}"] = reg.predict(original_X)
    df["residuals"] = df[y_col_name] - df[f"estimated_{y_col_name}"]

    # results coeficients
    dict_coef = dict(coef)
    dict_coef["intercept"] = dict_coef.pop("const")

    # Linear Trend values
    key_list = [
        "t",
        "delta_t",
    ]
    dict_coef_trend = {key: dict_coef.get(key) for key in key_list}
    intercept = dict_coef.get("intercept")

    X, _, original_index = _preprocess_data_to_fit_linear_model(
        df=df,
        outbreak_date=outbreak_date,
        date_lower_limit=date_lower_limit,
        date_upper_limit=date_upper_limit,
        y_col_name=y_col_name,
        x_col_names=key_list,
    )
    X = X.loc[original_index].values

    df["linear_trend"] = (
        np.matmul(X, np.array(list(dict_coef_trend.values()))) + intercept
    )

    # Linear Trend without `after_outbreak` and `delta_t` parameters
    key_list = [
        "t",
    ]
    dict_coef_trend2 = {key: dict_coef.get(key) for key in key_list}

    X, _, original_index = _preprocess_data_to_fit_linear_model(
        df=df,
        outbreak_date=outbreak_date,
        date_lower_limit=date_lower_limit,
        date_upper_limit=date_upper_limit,
        y_col_name=y_col_name,
        x_col_names=key_list,
    )

    X = X.loc[original_index].values

    df["expected_not_covid_linear_trend"] = (
        np.matmul(X, np.array(list(dict_coef_trend2.values()))) + intercept
    )

    if return_coef:
        return df, dict_coef, reg.summary(alpha=alpha)
    else:
        return df


def get_sa_data(
    conf_name: str,
    source: str,
    keep_only_positive: bool = False,
    keep_only_not_recurrent: bool = False,
) -> pd.DataFrame:
    """
    Function to import results of SA stays from different sources :
        - Text source rule based ("text_rule_based")
        - Text source ML based ("text_ml")
        - ICD10 claim data ("icd10")

    Parameters
    ----------
    conf_name: str,
        Name of the configuration file
    source; str,
        origin of data, one of {'text_rule_based','text_ml', 'icd10'}
    keep_only_positive: bool, default=False,
        If True, it returns just the visits identified as positive cases
    keep_only_not_recurrent: bool, default=False,
        If True, it returns not recurrent positive stays
    Returns
    -------
    df: pd.DataFrame,
        One line per stay. The column `source` shows the origin of the line.
    """

    assert source in {"text_rule_based", "icd10", "text_ml"}
    parameters = get_conf(conf_name)

    if source == "icd10":
        # Import SA stays identified by claim data (icd10 codes)
        path = os.path.expanduser(
            f"~/cse_210013/data/{conf_name}/stay_classification_claim_data_{conf_name}"
        )
        df = pd.read_pickle(path)

    # Import SA stays identified by text data (nlp-ml)
    if source == "text_ml":
        path = os.path.expanduser(
            f"~/cse_210013/data/{conf_name}/stay_classification_text_data_{conf_name}"
        )
        df = pd.read_pickle(path)

    if source == "text_rule_based":
        df = stay_classification_text_data(
            conf_name=conf_name,
            text_classification_method="rule_based",
        )

    if keep_only_positive:
        df = df.loc[df.positive_visit].copy()

    # Filter out unknown stays without unknown sex or age info
    df = filter_out_unknown_age_sex(df)

    # Keep stays only beetween study dates
    date_upper_limit = parameters["date_upper_limit"]
    date_lower_limit = parameters["date_from"]
    df = df.loc[(date_lower_limit <= df.date) & (df.date < date_upper_limit)]

    # Filter out recurrents stays
    if keep_only_not_recurrent:
        df = df.loc[np.logical_not(df.recurrent_visit)]

    # Sort by date
    df.sort_values("visit_start_date", inplace=True)

    # Reset index
    df.reset_index(inplace=True, drop=True)

    return df


def _groupby_for_plot(df: pd.DataFrame, gb_cols: list) -> pd.DataFrame:
    """Function that aggreagates data by `gb_cols`

    Aggregation
    ----------
    - ir: incidence rate
    - death: sum
    - duration_seconds: mean
    - has_history: sum
    - regex_rf : sum

    Parameters
    ----------
    df : pd.DataFrame
        the data to plot
    gb_cols : list
        columns to groupby

    Returns
    -------
    pd.DataFrame
        a df ready for plot with aggregated data
    """
    if ("stay_weight" in df.columns) and ("has_history" in df.columns):
        df_plot = df.groupby(gb_cols).agg(
            n_sa=("encounter_num", "count"),
            death=("death", "sum"),
            mean_duration=("duration_seconds", "mean"),
            has_history=("has_history", "sum"),
            weighted_n_sa=("stay_weight", "sum"),
        )
        df_plot["has_history_ratio"] = np.divide(
            df_plot["has_history"], df_plot["n_sa"]
        )
    else:
        df_plot = df.groupby(gb_cols).agg(
            n_sa=("encounter_num", "count"),
            death=("death", "sum"),
            mean_duration=("duration_seconds", "mean"),
        )

    if list(regex_rf.keys())[0] in df.columns:
        df_temp = df[gb_cols + list(regex_rf.keys())].groupby(gb_cols).sum()
        df_plot = df_plot.merge(df_temp, how="left", left_index=True, right_index=True)

        for key in regex_rf.keys():
            df_plot[f"{key}_ratio"] = np.divide(df_plot[key], df_plot["n_sa"])

    df_plot = df_plot.reset_index()
    df_plot.sort_values("date", inplace=True)

    df_plot["month"] = df_plot.date.dt.month
    df_plot["year"] = df_plot.date.dt.year
    df_plot["month_name"] = df_plot.date.dt.month_name()
    df_plot["death_ratio"] = np.divide(df_plot["death"], df_plot["n_sa"])

    df_plot["mean_duration"] = df_plot["mean_duration"] / seconds_per_day

    return df_plot


def get_data_for_km(
    df: pd.DataFrame,
    date_limit_km: str,
    covid_date: str,
    severity_event: str = "exists_end_of_stay",
    event_date="visit_end_date",
) -> Dict[str, pd.Series]:
    """Get data for the Kaplan Meier plot

    Parameters
    ----------
    df : pd.DataFrame
        df of type get_sa_data. For ex: `df_not_recurrent`
    date_limit_km : str
        date to consider as the "end" for the KM.
        (to replace None values for `visit_end_date`)
    covid_date : str
        date of the COVID19 outbreak used to divide in two cohorts

    Returns
    -------
    Dict[str, pd.Series]
        A dictionary with durations (T_1 and T_2) and boolean vectors (E_1 and E_2)
        indicating the presence of the event (exit of hospital)
    """
    df["km_end_date"] = df[event_date]

    df["km_end_date"].fillna(pd.to_datetime(date_limit_km), inplace=True)

    df["km_duration"] = df["km_end_date"] - df["visit_start_date"]
    df["km_duration"] = df.km_duration.dt.total_seconds()
    seconds_per_day = 60 * 60 * 24
    df["km_duration"] = df["km_duration"] / seconds_per_day

    if severity_event == "exists_end_of_stay":
        df[severity_event] = df.visit_end_date.notna()

    if df[severity_event].dtype != bool:
        df[severity_event] = df[severity_event].astype(bool)

    T1 = df.loc[df.date < covid_date].km_duration
    T2 = df.loc[df.date >= covid_date].km_duration
    E1 = df.loc[df.date < covid_date, severity_event]
    E2 = df.loc[df.date >= covid_date, severity_event]

    return {"T_1": T1, "T_2": T2, "E_1": E1, "E_2": E2}


def fit_linear_model(
    df: pd.DataFrame,
    date_lower_limit: str,
    date_upper_limit: str,
    gb_cols: list = [],
    return_coef: bool = False,
    y_col_name: str = "n_sa",
    **kwargs,
) -> Union[Tuple[pd.DataFrame, Dict[str, float], Dict[str, Summary]], pd.DataFrame,]:
    """Auxiliary function to apply repeated steps (aggregate and fit linear model)

    Parameters
    ----------
    df : pd.DataFrame
        data result of applying the function `suicide_attempt.functions.preprocess_data`
    gb_cols : list, optional
        groupby columns, by default [].
        `sex_cd` should be before `age_cat`, for example: ["sex_cd", "age_cat", "date"]
    return_coef : bool, optional
        either to return the coeficients of the fitted model or not, by default False
    y_col_name : str
        Column name to fit the regression, by default "n_sa"
    date_lower_limit : str
        date of start of data availability
    date_upper_limit : str
        max date available (non inclusive)

    Returns
    -------
    Union[Tuple[pd.DataFrame, Dict[str, float], Dict[str, Summary]], pd.DataFrame]

    If return_coef is True:
        Tuple[pd.DataFrame, Dict[str:float],Dict[str, Summary]]

            - df result of pipe_processing
            - Dictionnary of Coefficients of Linear Regression
            - Summary (summary of fit)
    Else:
    pd.DataFrame
        - df result of pipe_processing
    """
    # Sanity check
    check_gb_cols(gb_cols)

    # Aggregate - Group by age, sex & date
    df_gpd = _groupby_for_plot(df, gb_cols=gb_cols)

    if "date" in gb_cols:
        gb_cols.remove("date")

    if len(gb_cols) == 0:
        gb_cols = None

    if gb_cols is None:
        df_out, dict_coef, reg_summary = fit_and_predict_linear_model_for_group(
            df_gpd,
            outbreak_date=covid_date,
            date_lower_limit=date_lower_limit,
            date_upper_limit=date_upper_limit,
            x_col_names=None,
            y_col_name=y_col_name,
        )
    else:
        results = df_gpd.groupby(gb_cols).apply(
            fit_and_predict_linear_model_for_group,
            **dict(
                outbreak_date=covid_date,
                date_lower_limit=date_lower_limit,
                date_upper_limit=date_upper_limit,
                return_coef=True,
                x_col_names=None,
                y_col_name=y_col_name,
            ),
        )

        df_out_list = []
        dict_coef = {}
        reg_summary = {}
        for t in results:
            df_out_list.append(t[0])
            cat = "-".join(t[0][gb_cols].iloc[0].values)
            dict_coef[cat] = t[1]
            reg_summary[cat] = t[2]
        df_out = pd.concat(df_out_list)

    if not return_coef:
        return df_out
    else:
        return df_out, dict_coef, reg_summary


def add_rf_data(df: pd.DataFrame, conf_name: str, **kwargs) -> pd.DataFrame:
    """Function to add columns related to the Risk Factors to SA_data

    Parameters
    ----------
    conf_name : str

    Returns
    -------
    pd.DataFrame,
        same as `get_sa_data` but with extra columns related to Risk factors
    """
    dfs_patient_context = {}
    for key in regex_rf.keys():  # it's possible to optimize computation of this step
        if key != "suicide":  # suicide has a different processing pipeline
            dfs_patient_context[key] = import_patient_rf_data(
                conf_name,
                subcats=[key],
                text_modifers=["negated", "hypothesis"],
            )

            df[key] = df.encounter_num.isin(dfs_patient_context[key].encounter_num)

    return df


def _get_value_and_confidence_interval(
    reg_dict: Dict[str, Any],
    index_name: str = "hospital_label",
    label: str = None,
    **kwargs,
) -> pd.DataFrame:
    results = {}
    for key in reg_dict:
        reg = reg_dict[key]

        results_as_html = reg.tables[1].as_html()
        s = pd.read_html(results_as_html, header=0, index_col=0)[0]
        s.rename(
            columns={"[0.025": "cil", "0.975]": "ciu", "coef": "delta_t"}, inplace=True
        )
        h = s.loc["delta_t", ["delta_t", "cil", "ciu"]].to_dict()
        results[key] = h

    dfresults = pd.DataFrame(results).T
    dfresults = dfresults.rename_axis(index=index_name).reset_index()

    dfresults["lower_delta"] = dfresults["delta_t"] - dfresults["cil"]
    dfresults["upper_delta"] = dfresults["ciu"] - dfresults["delta_t"]
    dfresults["label"] = label

    return dfresults


def fit_linear_model_and_get_trend_difference(
    df: pd.DataFrame,
    gb_cols: List[str],
    y_col_name: str = "n_sa",
    label: Optional[str] = None,
    split_cols: Optional[List[str]] = None,
    query: Optional[str] = None,
    save_args: Dict[str, Any] = dict(save=False),
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Function to groupby values, fit the interrupted seasonally adjusted linear model
     and retrieve values of the fit.

    Parameters
    ----------
    df : pd.DataFrame
        data result of applying the function `suicide_attempt.functions.preprocess_data`
    gb_cols : List[str]
        list of column names to groupby
    y_col_name : str, optional
        column name to fit as the dependent variable, by default "n_sa"
    label : Optional[str], optional
        label of the results, for example one of {'by_hosp', 'all_pop'}, by default None
    split_cols : Optional[List[str]], optional
        columns to split a concatenated index name, by default None
    query : Optional[str], optional
        whether to query the results to extract only 'a view' of trend_dif,
         for example "age_min=='8' & age_max=='17'& sex_cd=='W'" , by default None
    save_args : Dict[str, Any], optional
        save details, by default dict(save=False)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]
        _description_
    """

    df_plot, _, reg_summary = fit_linear_model(
        df,
        gb_cols=gb_cols,
        return_coef=True,
        y_col_name=y_col_name,
        **kwargs,
    )

    # Get values for eFigure by hospital
    if (label == "all_pop") & (type(reg_summary) != dict):
        reg_summary_dict = {"All Hospitals": reg_summary}
    else:
        reg_summary_dict = reg_summary

    # Extract results of trend difference and their confidence interval
    trend_dif = _get_value_and_confidence_interval(
        reg_summary_dict, label=label, **kwargs
    )

    if label == "all_pop":
        trend_dif["hospital_label"] = "All Hospitals"

    if split_cols is not None:
        trend_dif[split_cols] = trend_dif[kwargs["index_name"]].str.split(
            pat=r"(?<!\d)-|-(?!\d)", expand=True
        )

    if query is not None:
        trend_dif = trend_dif.query(query).copy()

    if split_cols is not None:
        trend_dif.drop(
            columns=["age_cat", "sex_cd", "sex_age", "group_cat"],
            inplace=True,
            errors="ignore",
        )

    if "hospital_label" in trend_dif.columns:
        # Add type of hospital (Paediatric / Adult / both)
        hospital_type = pd.DataFrame.from_dict(
            hospital_type_dict, orient="index", columns=["hospital_type"]
        )
        hospital_type.index.name = "hospital_label"
        hospital_type.reset_index(
            drop=False,
            inplace=True,
        )
        trend_dif = trend_dif.merge(
            hospital_type, how="left", validate="many_to_one", on="hospital_label"
        )

        # Add number of stays by hospital
        if y_col_name == "weighted_n_sa":
            agg_func = ("stay_weight", "sum")
        else:
            agg_func = ("encounter_num", "count")

        if query is None:
            n_stays = df.groupby("hospital_label", as_index=False).agg(n_stays=agg_func)
        else:
            n_stays = (
                df.query(query)
                .groupby("hospital_label", as_index=False)
                .agg(n_stays=agg_func)
            )
        n_stays["n_stays"] = (
            n_stays["n_stays"].round().astype(int)
        )  # We should specify this because of the sum of stay_weight

        trend_dif = trend_dif.merge(
            n_stays, how="left", validate="many_to_one", on="hospital_label"
        )

    if save_args["save"]:
        if type(reg_summary) == dict:
            for key in reg_summary.keys():
                save_file(
                    file=reg_summary[key].as_text(),
                    conf_name=save_args.get("conf_name"),
                    name=f"{save_args.get('name')}{key}.txt",
                )
        else:
            save_file(
                file=reg_summary.as_text(),
                conf_name=save_args.get("conf_name"),
                name=f"{save_args.get('name')}.txt",
            )

    return df_plot, reg_summary, trend_dif


def get_tables_for_analysis(
    conf_name: str, save: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Get different df ready for the plots

    Parameters
    ----------
    conf_name : str

    Returns
    -------
    Dict[str,pd.DataFrame]
    """
    # Parameters
    parameters = get_conf(conf_name)

    # Import data
    df_text_ml = get_sa_data(
        conf_name=conf_name,
        source="text_ml",
        keep_only_positive=True,
        keep_only_not_recurrent=True,
    )
    df_text_ml = add_rf_data(df_text_ml, conf_name=conf_name)

    df_icd10 = get_sa_data(
        conf_name=conf_name,
        source="icd10",
        keep_only_positive=True,
        keep_only_not_recurrent=True,
    )

    # Aggregate - Group by age, sex & date Claim Source
    (
        df_plot_claim_source_sex_age,
        reg_summary_claim_source_sex_age,
        trend_dif_claim_source_w_8_17,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_icd10,
        gb_cols=["sex_cd", "age_cat", "date"],
        label="all_pop",
        split_cols=["sex_cd", "age_cat"],
        query="age_cat=='8-17' & sex_cd=='W'",
        index_name="sex_age",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by sex & date Texts Source
    (df_plot_texts_source_sex, _, reg_summary_texts_source_sex,) = fit_linear_model(
        df=df_text_ml,
        gb_cols=["sex_cd", "date"],
        return_coef=True,
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by sex & date Claim Source
    (df_plot_claim_source_sex, _, reg_summary_claim_source_sex,) = fit_linear_model(
        df=df_icd10,
        gb_cols=["sex_cd", "date"],
        return_coef=True,
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Sensitivity analysis - rule based NLP classification
    df_text_rule_based = get_sa_data(
        conf_name=conf_name,
        source="text_rule_based",
        keep_only_positive=True,
        keep_only_not_recurrent=True,
    )
    df_text_rule_based = add_rf_data(
        df_text_rule_based,
        conf_name=conf_name,
    )

    (
        df_plot_texts_source_all_pop_rule_based,
        reg_summary_texts_source_all_pop_rule_based,
        trend_dif_texts_source_all_pop_rule_based,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_rule_based,
        gb_cols=["date"],
        label="all_pop",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    (
        df_plot_texts_source_all_pop_hosp_rule_based,
        reg_summary_texts_source_all_pop_hosp_rule_based,
        trend_dif_texts_source_all_pop_hosp_rule_based,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_rule_based,
        gb_cols=["date", "hospital_label"],
        label="by_hosp",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    (
        df_plot_texts_source_sex_age_rule_based,
        reg_summary_texts_source_sex_age_rule_based,
        trend_dif_texts_source_w_8_17_rule_based,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_rule_based,
        gb_cols=["sex_cd", "age_cat", "date"],
        label="all_pop",
        split_cols=["sex_cd", "age_cat"],
        query="age_cat=='8-17' & sex_cd=='W'",
        index_name="sex_age",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    (
        df_plot_texts_source_sex_age_hosp_rule_based,
        reg_summary_texts_source_sex_age_hosp_rule_based,
        trend_dif_texts_source_w_8_17_hosp_rule_based,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_rule_based,
        gb_cols=["sex_cd", "age_cat", "date", "hospital_label"],
        label="by_hosp",
        split_cols=["sex_cd", "age_cat", "hospital_label"],
        query="age_cat=='8-17' & sex_cd=='W'",  # age='8-17', sex='W'
        index_name="group_cat",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    (_, _, reg_summary_texts_source_sex_rule_based,) = fit_linear_model(
        df_text_rule_based,
        gb_cols=["date", "sex_cd"],
        return_coef=True,
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date - Overall population - Texts source
    (
        df_plot_texts_source_all_pop,
        reg_summary_texts_source_all_pop,
        trend_dif_texts_source_all_pop,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["date"],
        label="all_pop",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date - Overall population - Texts source (date, hospital)
    (
        df_plot_texts_source_all_pop_hosp,
        _,
        trend_dif_texts_source_all_pop_hosp,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["date", "hospital_label"],
        label="by_hosp",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date - Overall population - Texts source - rescaled
    (
        df_plot_texts_source_all_pop_rescaled,
        reg_summary_texts_source_all_pop_rescaled,
        trend_dif_texts_source_all_pop_rescaled,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["date"],
        y_col_name="weighted_n_sa",
        label="all_pop",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date - Overall population -
    # Texts source (date, hospital) - rescaled
    (
        df_plot_texts_source_all_pop_hosp_rescaled,
        _,
        trend_dif_texts_source_all_pop_hosp_rescaled,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["date", "hospital_label"],
        y_col_name="weighted_n_sa",
        label="by_hosp",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by age, sex & date Texts Source
    (
        df_plot_texts_source_sex_age,
        reg_summary_texts_source_sex_age,
        trend_dif_texts_source_w_8_17,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["sex_cd", "age_cat", "date"],
        label="all_pop",
        split_cols=["sex_cd", "age_cat"],
        query="age_cat=='8-17' & sex_cd=='W'",
        index_name="sex_age",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by age, sex, date, Texts Source & Hospital
    (
        df_plot_texts_source_sex_age_hosp,
        _,
        trend_dif_texts_source_w_8_17_hosp,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["sex_cd", "age_cat", "date", "hospital_label"],
        label="by_hosp",
        split_cols=["sex_cd", "age_cat", "hospital_label"],
        query="age_cat=='8-17' & sex_cd=='W'",  # age='8-17', sex='W'
        index_name="group_cat",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by age, sex, date, Claim Source & Hospital
    (
        df_plot_claim_source_sex_age_hosp,
        _,
        trend_dif_claim_source_w_8_17_hosp,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_icd10,
        gb_cols=["sex_cd", "age_cat", "date", "hospital_label"],
        label="by_hosp",
        split_cols=["sex_cd", "age_cat", "hospital_label"],
        query="age_cat=='8-17' & sex_cd=='W'",  # age='8-17', sex='W'
        index_name="group_cat",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by age, sex & date Texts Source - rescaled
    (
        df_plot_texts_source_sex_age_rescaled,
        reg_summary_texts_source_sex_age_rescaled,
        _,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["sex_cd", "age_cat", "date"],
        label="all_pop",
        y_col_name="weighted_n_sa",
        split_cols=["sex_cd", "age_cat"],
        query="age_cat=='8-17' & sex_cd=='W'",
        index_name="sex_age",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    (_, _, reg_summary_texts_source_sex_rescaled,) = fit_linear_model(
        df_text_ml,
        gb_cols=["date", "sex_cd"],
        return_coef=True,
        y_col_name="weighted_n_sa",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by age, sex, date, Texts Source & Hospital
    (
        df_plot_texts_source_sex_age_hosp_rescaled,
        _,
        trend_dif_texts_source_w_8_17_hosp_rescaled,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_text_ml,
        gb_cols=["sex_cd", "age_cat", "date", "hospital_label"],
        label="by_hosp",
        y_col_name="weighted_n_sa",
        split_cols=["sex_cd", "age_cat", "hospital_label"],
        query="age_cat=='8-17' & sex_cd=='W'",  # age='8-17', sex='W'
        index_name="group_cat",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date - Overall population - CLAIM source
    (
        df_plot_claim_source_all_pop,
        reg_summary_claim_source_all_pop,
        _,
    ) = fit_linear_model_and_get_trend_difference(
        df_icd10,
        gb_cols=["date"],
        label="all_pop",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Aggregate - Group by date, hospital - Overall population - CLAIM source
    (
        df_plot_claim_source_all_pop_hosp,
        _,
        trend_dif_claim_source_all_pop_hosp,
    ) = fit_linear_model_and_get_trend_difference(
        df=df_icd10,
        gb_cols=["date", "hospital_label"],
        label="by_hosp",
        date_lower_limit=parameters["date_from"],
        date_upper_limit=parameters["date_upper_limit"],
    )

    # Get data for Kaplan Meier
    km_data = get_data_for_km(
        df_text_ml,
        date_limit_km=parameters["date_upper_limit"],
        covid_date=covid_date,
        severity_event="exists_end_of_stay",
    )
    km_data_death = get_data_for_km(
        df_text_ml,
        date_limit_km=parameters["date_upper_limit"],
        covid_date=covid_date,
        severity_event="death",
    )

    table1 = get_table1(
        reg_summary_sex_age=reg_summary_texts_source_sex_age,
        reg_summary_sex=reg_summary_texts_source_sex,
        reg_summary_all_pop=reg_summary_texts_source_all_pop,
        df_sex_age=df_plot_texts_source_sex_age,
    )
    if save:
        save_file(
            file=table1.to_csv(index=False),
            conf_name=conf_name,
            name="table1.csv",
        )

    sm_table1_claim = get_table1(
        reg_summary_sex_age=reg_summary_claim_source_sex_age,
        reg_summary_sex=reg_summary_claim_source_sex,
        reg_summary_all_pop=reg_summary_claim_source_all_pop,
        df_sex_age=df_plot_claim_source_sex_age,
    )
    if save:
        save_file(
            file=sm_table1_claim.to_csv(index=False),
            conf_name=conf_name,
            name="eTable2_claim.csv",
        )

    sm_table1_rule_based = get_table1(
        reg_summary_sex_age=reg_summary_texts_source_sex_age_rule_based,
        reg_summary_sex=reg_summary_texts_source_sex_rule_based,
        reg_summary_all_pop=reg_summary_texts_source_all_pop_rule_based,
        df_sex_age=df_plot_texts_source_sex_age_rule_based,
    )
    if save:
        save_file(
            file=sm_table1_rule_based.to_csv(index=False),
            conf_name=conf_name,
            name="eTable1_rule_based.csv",
        )

    sm_table1_rescaled = get_table1(
        reg_summary_sex_age=reg_summary_texts_source_sex_age_rescaled,
        reg_summary_sex=reg_summary_texts_source_sex_rescaled,
        reg_summary_all_pop=reg_summary_texts_source_all_pop_rescaled,
        df_sex_age=df_plot_texts_source_sex_age_rescaled,
        y_col="weighted_n_sa",
    )
    if save:
        save_file(
            file=sm_table1_rescaled.to_csv(index=False),
            conf_name=conf_name,
            name="eTable3_rescaled.csv",
        )

    rf_melt_all_pop = df_plot_texts_source_all_pop.melt(
        id_vars="date",
        var_name="rf",
        value_name="prevalence",
        value_vars=[
            "sexual_violence_ratio",
            "domestic_violence_ratio",
            "physical_violence_ratio",
            "social_isolation_ratio",
            "has_history_ratio",
        ],
    )

    rf_melt_by_sex = df_plot_texts_source_sex.melt(
        id_vars=["date", "sex_cd"],
        var_name="rf",
        value_name="prevalence",
        value_vars=[
            "sexual_violence_ratio",
            "domestic_violence_ratio",
            "physical_violence_ratio",
            "social_isolation_ratio",
            "has_history_ratio",
        ],
    )

    # Table 3 of the article
    table3 = get_table3(df_plot_texts_source_sex)
    if save:
        save_file(
            file=table3.to_csv(index=True),
            conf_name=conf_name,
            name="table3.csv",
        )

    # Data for SA modality plot
    dict_modality_sa = get_modality_sa(
        df_text_ml, generic_and_others=set(unknown_and_other_forms)
    )

    # Data for RF
    df_plot_rf = process_for_rf_plot(
        df_plot_texts_source_sex, df_plot_texts_source_all_pop
    )

    return dict(
        df_plot_claim_source_sex_age=df_plot_claim_source_sex_age,
        df_plot_texts_source_sex_age=df_plot_texts_source_sex_age,
        df_plot_texts_source_all_pop=df_plot_texts_source_all_pop,
        km_data=km_data,
        df_text_ml=df_text_ml,
        df_icd10=df_icd10,
        df_plot_texts_source_all_pop_hosp=df_plot_texts_source_all_pop_hosp,
        df_plot_claim_source_all_pop_hosp=df_plot_claim_source_all_pop_hosp,
        df_plot_claim_source_all_pop=df_plot_claim_source_all_pop,
        df_plot_texts_source_all_pop_rule_based=df_plot_texts_source_all_pop_rule_based,
        df_plot_texts_source_sex_age_rule_based=df_plot_texts_source_sex_age_rule_based,
        reg_summary_texts_source_sex_age=reg_summary_texts_source_sex_age,
        reg_summary_texts_source_all_pop=reg_summary_texts_source_all_pop,
        reg_summary_texts_source_sex=reg_summary_texts_source_sex,
        table1=table1,
        df_plot_texts_source_sex_age_hosp=df_plot_texts_source_sex_age_hosp,
        df_plot_texts_source_sex=df_plot_texts_source_sex,
        rf_melt_all_pop=rf_melt_all_pop,
        rf_melt_by_sex=rf_melt_by_sex,
        dict_modality_sa=dict_modality_sa,
        df_plot_texts_source_all_pop_rescaled=df_plot_texts_source_all_pop_rescaled,
        df_plot_texts_source_all_pop_hosp_rescaled=df_plot_texts_source_all_pop_hosp_rescaled,  # noqa: E501
        df_plot_claim_source_sex_age_hosp=df_plot_claim_source_sex_age_hosp,
        df_plot_texts_source_sex_age_rescaled=df_plot_texts_source_sex_age_rescaled,
        df_plot_texts_source_sex_age_hosp_rescaled=df_plot_texts_source_sex_age_hosp_rescaled,  # noqa: E501
        km_data_death=km_data_death,
        trend_dif_claim_source_all_pop_hosp=trend_dif_claim_source_all_pop_hosp,
        trend_dif_claim_source_w_8_17_hosp=trend_dif_claim_source_w_8_17_hosp,
        trend_dif_texts_source_all_pop_hosp_rescaled=trend_dif_texts_source_all_pop_hosp_rescaled,  # noqa: E501
        trend_dif_texts_source_w_8_17_hosp=trend_dif_texts_source_w_8_17_hosp,
        trend_dif_texts_source_w_8_17_hosp_rescaled=trend_dif_texts_source_w_8_17_hosp_rescaled,  # noqa: E501
        trend_dif_texts_source_all_pop_hosp_rule_based=trend_dif_texts_source_all_pop_hosp_rule_based,  # noqa: E501
        trend_dif_texts_source_w_8_17_hosp_rule_based=trend_dif_texts_source_w_8_17_hosp_rule_based,  # noqa: E501
        trend_dif_texts_source_all_pop_hosp=trend_dif_texts_source_all_pop_hosp,
        df_plot_rf=df_plot_rf,
    )


def get_modality_sa(
    df: pd.DataFrame, generic_and_others: set = {"SA", "Suicide attempt", "Other forms"}
) -> Dict[str, pd.DataFrame]:

    # Explode df by type of std_positive_lexical_variant
    dfe = df[["encounter_num", "std_positive_lexical_variants"]].explode(
        "std_positive_lexical_variants"
    )

    # Pivot table, one line by stay, one column by std_lexical variant
    table = pd.pivot_table(
        dfe,
        values="std_positive_lexical_variants",
        index=[
            "encounter_num",
        ],
        columns=["std_positive_lexical_variants"],
        aggfunc=len,
        fill_value=0,
    )

    # Select all columns different from `generic_and_others`
    # because these are considered generic mentions.
    cols = set(table.columns).difference(generic_and_others)
    table_not_ts = table[cols].copy()

    # Total of positive mentions by stay
    table_not_ts["total"] = table_not_ts.sum(axis=1)

    # Divide all columns by the total of positive mention by row
    cols = set(table_not_ts.columns).difference({"total"})
    table_not_ts_2 = table_not_ts[cols].div(table_not_ts["total"], axis=0)

    # For those stays with only `generic_and_others` mentions, count 1 for Unknown
    table_not_ts_2["Unknown & Other forms"] = np.where(table_not_ts["total"] == 0, 1, 0)
    cols.add("Unknown & Other forms")

    # Add date, sex and age information
    table_not_ts_3 = table_not_ts_2.merge(
        df[["encounter_num", "date", "sex_cd", "age_cat"]],
        left_index=True,
        right_on="encounter_num",
        how="left",
        validate="one_to_one",
    )

    # Group by gb_cols and get totals by date
    gb_cols_dict = {"Overall population": ["date"], "by_sex": ["date", "sex_cd"]}
    results = {}

    for item in gb_cols_dict.items():
        key, gb_cols = item

        df_plot = table_not_ts_3.groupby(gb_cols, as_index=False, observed=True)[
            list(cols)
        ].sum()
        df_plot["total"] = df_plot[list(cols)].sum(axis=1)

        # Get ratio by modality for each date
        df_plot[list(cols)] = df_plot[list(cols)].div(df_plot["total"], axis=0)
        del df_plot["total"]

        results[key] = df_plot.copy()

    if "by_sex" in results.keys():
        results["Male"] = results["by_sex"].query("sex_cd=='M'")
        results["Female"] = results["by_sex"].query("sex_cd=='W'")
        results.pop("by_sex")

    # sort columns by total sum
    all_cols = set(results["Overall population"].columns)
    not_labels = set(["date", "sex_cd", "age_cat", "Unknown & Other forms"])
    cols = all_cols - not_labels
    cols_list = list(cols)

    sorted_cols = (
        results["Overall population"][cols_list].sum().sort_values().index.to_list()
    )

    for key, data in results.items():
        all_cols = set(data.columns)
        remaining = list(all_cols - set(sorted_cols))

        new_order = list(sorted_cols) + remaining

        results[key] = data[new_order]

    return results


class GetSAStaysNLP:
    def __init__(self, conf_name, test_set=False):
        self.parameters = get_conf(conf_name)
        self.conf_name = conf_name
        self.test_set = test_set

    def _get_rb_stays(self):
        stays = get_sa_data(
            conf_name=self.conf_name,
            keep_only_positive=False,
            source="text_rule_based",
        )

        stays.rename(
            columns={
                "nlp_positive": "rb_positive",
                "recurrent_visit": "recurrent_visit_rb",
            },
            inplace=True,
        )

        stays = self._preprocess(stays)

        return stays

    def _get_ml_stays(self):
        stays = get_sa_data(self.conf_name, source="text_ml", keep_only_positive=False)
        stays = add_rf_data(
            stays,
            conf_name=self.conf_name,
        )
        stays.rename(
            columns={
                "nlp_positive": "ml_positive",
                "recurrent_visit": "recurrent_visit_ml",
            },
            inplace=True,
        )

        stays = self._preprocess(stays)
        return stays

    def _preprocess(self, df):
        if self.test_set:
            hospitals_test = self.parameters["hospitals_test"]  # noqa: F841
            df = df.query("hospital_label.isin(@hospitals_test)")

        # Filter by date
        df = df.loc[df.date < self.parameters["date_upper_limit"]]

        # Add column `after_covid_outbreak`
        df["after_covid_outbreak"] = df.date >= covid_date

        # Explode column of lists (list are always of size 1)
        df = df.explode(column="note_ids")
        df.rename(columns={"note_ids": "note_id"}, inplace=True)

        return df

    def get_nlp_stays(self):
        rb = self._get_rb_stays()
        ml = self._get_ml_stays()
        cols = [
            "ml_positive",
            "recurrent_visit_ml",
            "encounter_num",
            "note_id",
            "has_history",
            "age_cat",
            "age",
            "date",
            "hospital_label",
            "sexual_violence",
            "domestic_violence",
            "physical_violence",
            "social_isolation",
            "after_covid_outbreak",
        ]
        df = rb[["encounter_num", "rb_positive", "recurrent_visit_rb"]].merge(
            ml[cols],
            on="encounter_num",
            how="outer",
            validate="one_to_one",
        )

        assert df.encounter_num.is_unique

        df["note_id_dedup"] = df["note_id"]

        return df


class ValidationMetrics:
    def __init__(
        self,
        annotation_subset,
        annotators,
        data_to_validate=None,
        label_y_pred=None,
    ):
        self.annotation_subset = annotation_subset
        self.annotators = annotators

        self.save_path = os.path.expanduser(
            "~/cse_210013/data/annotation/validation/annotated/{annotation_subset}/{annotation_subset}_{annotator}.pickle"  # noqa: E501
        )

        self.save_path_supp = os.path.expanduser(
            "~/cse_210013/data/annotation/validation/annotated/{annotation_subset}/{annotation_subset}_{annotator}_supplementary.pickle"  # noqa: E501
        )

        self.ref = {
            "SA-ML": {
                "label_y_true": "SA-ML Gold Standard",
                "label_y_pred": "SA-ML stay",
            },
            "SA-RB": {
                "label_y_true": "SA-RB Gold Standard",
                "label_y_pred": "SA-RB stay",
            },
            "RF": {"label_y_true": "RF Gold Standard", "label_y_pred": "RF stay"},
        }

        assert annotation_subset in self.ref.keys()

        self.label_y_true = self.ref[annotation_subset]["label_y_true"]
        if label_y_pred is None:
            self.label_y_pred = self.ref[annotation_subset]["label_y_pred"]
            self._label_y_pred_annotations = self.ref[annotation_subset]["label_y_pred"]
        else:
            self.label_y_pred = label_y_pred
            self._label_y_pred_annotations = self.ref[annotation_subset]["label_y_pred"]

        self.data_to_validate = data_to_validate
        if data_to_validate is not None:
            self.data_to_validate = self.data_to_validate.loc[
                self.data_to_validate[self.label_y_pred]
            ]

        self.cols = [
            "note_id",
            "note_id_dedup",
            "visit_start_date",
            "encounter_num",
            "patient_num",
            "after_covid_outbreak",
            "annotation_subset",
            "annotator",
            "concept_cd",
            self._label_y_pred_annotations,
            self.label_y_true,
            "Remarque",
            "supplementary",
            "note_text",
        ]

        # Posterior correction of annotated visits with inter-annotator disagreement
        self.conflicts = {
            "SA-ML": [
                ("-2414206652542356459", "vincent"),
                ("6156757603362854527", "vincent"),
            ],
            "SA-RB": [("-4170409160141682184", "vincent")],
            "RF": [],
        }

        if self.annotation_subset == "RF":
            self.cols.append("rf_name")

        self.periods = {
            False: "Pre-pandemic (95%CI, No. annotated records)",
            True: "Post-pandemic (95%CI, No. annotated records)",
        }

    def read_annotations_indiv(self, annotator: str, supplementary: bool):
        if not supplementary:
            path = self.save_path.format(
                annotation_subset=self.annotation_subset,
                annotator=annotator,
            )
        else:
            path = self.save_path_supp.format(
                annotation_subset=self.annotation_subset,
                annotator=annotator,
            )

        if os.path.exists(path):
            df = pd.read_pickle(path)
            df = df.loc[df.vu].copy()
            if self.annotation_subset != "RF":
                df = extract_global_labels(df)

            df.reset_index(inplace=True, drop=True)
            df["supplementary"] = supplementary
            df[self.label_y_true] = df[self.label_y_true].astype(bool)
            df[self._label_y_pred_annotations] = df[
                self._label_y_pred_annotations
            ].astype(bool)

            if self.annotation_subset == "RF":
                idx = df.query("Remarque in ( 'EL','EK')").index
                df.drop(index=idx, inplace=True)

            df.drop_duplicates("note_id_dedup", inplace=True)

            return df[self.cols]

    def get_annotations(self):
        dfs = []
        for annotator in self.annotators:
            for supplementary in [False, True]:
                df_indiv = self.read_annotations_indiv(annotator, supplementary)
                dfs.append(df_indiv)
        df = pd.concat(dfs)

        n_annotators = df.groupby("note_id_dedup", as_index=False).agg(
            n_annotator=("annotator", "nunique")
        )
        df = df.merge(
            n_annotators, on="note_id_dedup", how="left", validate="many_to_one"
        )

        disagreements = (
            df.query("n_annotator>1")
            .groupby("note_id_dedup", as_index=False)
            .agg(disagreement=(self.label_y_true, self.f_disagreement))
        )

        df = df.merge(
            disagreements, on="note_id_dedup", how="left", validate="many_to_one"
        )

        # Number of lines
        self.n_total = df.note_id_dedup.nunique()

        df.reset_index(inplace=True, drop=True)
        return df

    def f_disagreement(self, x):
        if len(set(x)) > 1:
            return True
        else:
            return False

    def disambiguate_conflicts(self, annotations):
        drop_index_series = pd.Index([])
        for note_id_dedup, annotator in self.conflicts[self.annotation_subset]:
            idx = annotations.query(
                f"note_id_dedup=='{note_id_dedup}' & annotator=='{annotator}'"
            ).index
            drop_index_series = drop_index_series.union(idx)

        annotations.drop(drop_index_series, inplace=True)

    def get_data_for_metric(self):
        annotations = self.get_annotations()
        self.disambiguate_conflicts(annotations)
        # annotations = annotations.query("disagreement != True").copy()
        annotations.drop_duplicates(subset=["note_id_dedup"], inplace=True)

        # Merge data if a df is passed
        if self.data_to_validate is not None:
            annotations.drop(columns=[self._label_y_pred_annotations], inplace=True)
            annotations = annotations.merge(
                self.data_to_validate[["note_id_dedup", self.label_y_pred]],
                on="note_id_dedup",
                how="inner",
                validate="one_to_one",
            )

        # Support after drop desagreement
        self.support = annotations.note_id_dedup.nunique()
        self.discarded = self.n_total - self.support
        assert self.discarded >= 0
        assert annotations.note_id_dedup.is_unique

        return annotations

    @staticmethod
    def _precision(y_true, y_pred):
        ppv = precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="binary",
        )

        return ppv

    def _confusion_matrix(self, y_true, y_pred, as_series=False):
        tn, fp, fn, tp = confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=[False, True]
        ).ravel()
        if as_series:
            return pd.Series([tn, fp, fn, tp])
        else:
            return tn, fp, fn, tp

    def _proportion_confint(self, y_true, y_pred):
        tn, fp, fn, tp = self._confusion_matrix(y_true, y_pred)
        confint = proportion_confint(count=tp, nobs=tp + fp, method="wilson")
        return confint

    def precision(self):
        annotations = self.get_data_for_metric()
        y_true = annotations[self.label_y_true]
        y_pred = annotations[self.label_y_pred]

        ppv = self._precision(y_true=y_true, y_pred=y_pred)

        return ppv

    def _precision_by_group(self, annotations, group, label_pred=None):
        if label_pred is None:
            label_pred = self.label_y_pred

        ppv = annotations.groupby(group).apply(
            lambda x: self._precision(y_true=x[self.label_y_true], y_pred=x[label_pred])
        )

        support = annotations.groupby(group).size()

        confint = annotations.groupby(group).apply(
            lambda x: self._proportion_confint(
                y_true=x[self.label_y_true], y_pred=x[label_pred]
            )
        )

        self.cm = annotations.groupby(group).apply(
            lambda x: self._confusion_matrix(
                y_true=x[self.label_y_true], y_pred=x[label_pred], as_series=True
            )
        )
        self.cm.columns = ["tn", "fp", "fn", "tp"]

        return ppv, support, confint

    def precision_by_period(self):
        annotations = self.get_data_for_metric()
        return self._precision_by_group(annotations, group=["after_covid_outbreak"])

    def precision_by_rf(self):
        annotations = self.get_data_for_metric()
        return self._precision_by_group(
            annotations,
            group=[
                "rf_name",
                "after_covid_outbreak",
            ],
        )

    def _print_output_summary(self, ppv, support, confint):

        print("Precision global:", self.precision())
        print("Total examples", self.n_total)
        print("Support", self.support)
        print("Discarded", self.discarded)

        print("\n### Metrics ###")
        results = {}
        for period in [False, True]:

            output = """# After covid outbreak: {period}\n PPV: {ppv:.2f} ({confint_lower:.2f} - {confint_upper:.2f}, {support})""".format(  # noqa: E501
                period=period,
                ppv=ppv.loc[period],
                confint_lower=confint.loc[period][0],
                confint_upper=confint.loc[period][1],
                support=support.loc[period],
            )

            results[
                self.periods[period]
            ] = """{ppv:.2f} ({confint_lower:.2f} - {confint_upper:.2f}, {support})""".format(  # noqa: E501
                ppv=ppv.loc[period],
                confint_lower=confint.loc[period][0],
                confint_upper=confint.loc[period][1],
                support=support.loc[period],
            )
            print(output)
            print("\n")
        return results

    def summary(self):
        if self.annotation_subset == "RF":
            _ppv, _support, _confint = self.precision_by_rf()
            results = {}
            for rf in _ppv.index.get_level_values(0).unique():
                ppv = _ppv.loc[rf]
                support = _support.loc[rf]
                confint = _confint.loc[rf]
                print(f"########## {rf} ###########")
                results[rf] = self._print_output_summary(ppv, support, confint)
        else:
            ppv, support, confint = self.precision_by_period()
            results = self._print_output_summary(ppv, support, confint)

        return results

    def get_disagreements(self):
        df = self.get_annotations()
        disagreements = df.query("disagreement==True").copy()
        disagreements.sort_values(["note_id_dedup", "annotator"], inplace=True)

        return disagreements

    def get_shared(self):
        assert (
            len(self.annotators) == 2
        ), "The number of annotators should be 2 for this cohen's kappa implementation"
        annotations = self.get_annotations()
        shared_long = annotations.query("n_annotator >1")

        if self.annotation_subset != "RF":

            shared = shared_long.pivot(
                index="note_id_dedup", columns="annotator", values=self.label_y_true
            )
        else:
            shared = shared_long.pivot(
                index=["note_id_dedup", "rf_name"],
                columns="annotator",
                values=self.label_y_true,
            )

            shared.reset_index(level=1, inplace=True)

        self.shared = shared
        return shared

    def compute_cohen_kappa(self):
        if self.annotation_subset != "RF":
            shared = self.get_shared()
            ck = cohen_kappa_score(
                shared[self.annotators[0]],
                shared[self.annotators[1]],
                labels=[False, True],
            )
            n_shared = len(shared)
        else:
            shared = self.get_shared()

            ck = shared.groupby("rf_name").apply(
                lambda x: cohen_kappa_score(
                    x[self.annotators[0]], x[self.annotators[1]], labels=[False, True]
                ),
            )

            n_shared = shared.groupby("rf_name").size()

            distinct_values = shared.groupby("rf_name").apply(
                lambda x: np.unique(x[self.annotators].values)
            )
            print("Distinct values\n", distinct_values)

        return ck, n_shared

    def get_cross_table(self, shared):
        crosstab = pd.crosstab(
            shared[self.annotators[0]], shared[self.annotators[1]], margins=True
        )
        return crosstab

    def _compute_agreement(self, crosstab, _class):
        if _class in crosstab.index:
            f1 = crosstab.loc["All", _class]
            g1 = crosstab.loc[_class, "All"]
            a = crosstab.loc[_class, _class]
            p = 2 * a / (f1 + g1)
            return p
        else:
            return None

    def compute_agreement(self):
        if self.annotation_subset != "RF":
            shared = self.get_shared()
            crosstab = self.get_cross_table(shared)
            ppos = self._compute_agreement(crosstab, True)
            pneg = self._compute_agreement(crosstab, False)

        else:
            shared = self.get_shared()

            crosstab = self.shared.groupby("rf_name").apply(
                lambda x: self.get_cross_table(x)
            )
            ppos = (
                crosstab.reset_index(level=0)
                .groupby("rf_name")
                .apply(lambda x: self._compute_agreement(x, True))
            )

            pneg = (
                crosstab.reset_index(level=0)
                .groupby("rf_name")
                .apply(lambda x: self._compute_agreement(x, False))
            )

        return dict(ppos=ppos, pneg=pneg)
