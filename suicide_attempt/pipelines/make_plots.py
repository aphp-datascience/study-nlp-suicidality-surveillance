import seaborn as sns
import typer

from suicide_attempt.functions.constants import covid_date
from suicide_attempt.functions.ehr_deployement import get_data_ehr_deployement
from suicide_attempt.functions.plot_utils import (
    forest_plot,
    main_plot,
    plot_kaplan_meier,
    plot_modality_sa,
    plot_ratio_docs_visit,
    plot_rf,
)
from suicide_attempt.functions.stats_utils import get_tables_for_analysis
from suicide_attempt.functions.utils import get_conf

sns.set()  # Setting


def make_plots(conf_name: str = typer.Argument(..., help="name of conf file")):
    """Pipeline to generate and save the plots.

    Parameters
    ----------
    conf_name : str
        name of configuration file
    """

    tables = get_tables_for_analysis(conf_name=conf_name)

    df_plot_claim_source_sex_age = tables["df_plot_claim_source_sex_age"]
    df_plot_texts_source_sex_age = tables["df_plot_texts_source_sex_age"]
    df_plot_texts_source_all_pop = tables["df_plot_texts_source_all_pop"]
    km_data = tables["km_data"]
    df_plot_claim_source_all_pop = tables["df_plot_claim_source_all_pop"]
    df_plot_texts_source_all_pop_rule_based = tables[
        "df_plot_texts_source_all_pop_rule_based"
    ]
    df_plot_texts_source_sex_age_rule_based = tables[
        "df_plot_texts_source_sex_age_rule_based"
    ]

    df_plot_texts_source_all_pop_rescaled = tables[
        "df_plot_texts_source_all_pop_rescaled"
    ]

    km_data_death = tables["km_data_death"]

    df_plot_texts_source_sex_age_rescaled = tables[
        "df_plot_texts_source_sex_age_rescaled"
    ]
    dict_modality_sa = tables["dict_modality_sa"]

    # Trend difference
    # Rule based
    trend_dif_texts_source_all_pop_hosp_rule_based = tables[
        "trend_dif_texts_source_all_pop_hosp_rule_based"
    ]
    trend_dif_texts_source_w_8_17_hosp_rule_based = tables[
        "trend_dif_texts_source_w_8_17_hosp_rule_based"
    ]
    # Claim source
    trend_dif_claim_source_all_pop_hosp = tables["trend_dif_claim_source_all_pop_hosp"]
    trend_dif_claim_source_w_8_17_hosp = tables["trend_dif_claim_source_w_8_17_hosp"]
    # Rescaled
    trend_dif_texts_source_w_8_17_hosp_rescaled = tables[
        "trend_dif_texts_source_w_8_17_hosp_rescaled"
    ]
    trend_dif_texts_source_all_pop_hosp_rescaled = tables[
        "trend_dif_texts_source_all_pop_hosp_rescaled"
    ]
    # Text source ML
    trend_dif_texts_source_all_pop_hosp = tables["trend_dif_texts_source_all_pop_hosp"]
    trend_dif_texts_source_w_8_17_hosp = tables["trend_dif_texts_source_w_8_17_hosp"]

    # RF
    df_plot_rf = tables["df_plot_rf"]

    # EHR Deployement
    parameters = get_conf(conf_name)
    hospital_list = parameters["hospitals_train"] + parameters["hospitals_test"]
    ehr_deployement1 = get_data_ehr_deployement(
        hospit_list=hospital_list[:7],
        date_max=parameters["date_upper_limit"],
        date_min=parameters["date_from"],
        name_file=parameters["ehr_deployement_file"],
    )
    ehr_deployement2 = get_data_ehr_deployement(
        hospit_list=hospital_list[7:],
        date_max=parameters["date_upper_limit"],
        date_min=parameters["date_from"],
        name_file=parameters["ehr_deployement_file"],
    )

    # PLOTS ############
    # Generate main plot of the article - figure1
    main_plot(
        df_plot_texts_source_sex_age,
        df_plot_texts_source_all_pop,
        filename="figure1",  # main
        conf_name=conf_name,
    )

    # Plot RF - figure2
    plot_rf(
        df_plot_rf,
        filename="figure2a",  # risk_factor_prevalence
        conf_name=conf_name,
    )

    plot_rf(
        df_plot_rf,
        filename="figure2a_bis",  # risk_factor_prevalence
        conf_name=conf_name,
        titles={
            "Sexual violence": "sexual_violence_ratio_smooth",
            "Domestic violence": "domestic_violence_ratio_smooth",
            "Physical violence": "physical_violence_ratio_smooth",
            "Social isolation": "social_isolation_ratio_smooth",
            "Suicide attempt history": "has_history_ratio_smooth",
        },
    )

    # eFigure 16: Per-hospital completeness of discharge summaries data
    plot_ratio_docs_visit(
        df=ehr_deployement1,
        conf_name=conf_name,
        filename="ehr_deployement/eFigure16a",  # ratio_docs_hospit_1
    )
    plot_ratio_docs_visit(
        df=ehr_deployement2,
        conf_name=conf_name,
        filename="ehr_deployement/eFigure16b",  # ratio_docs_hospit_2
    )

    # eFigure 1: Detected and modeled number of suicide attempts
    main_plot(
        df_plot_texts_source_sex_age,
        df_plot_texts_source_all_pop,
        filename="residuals/eFigure1",  # estimated_n_sa_season
        conf_name=conf_name,
        model_estimate_col="estimated_n_sa",
    )

    # eFigure 2: Residuals of the modeling - monthly numbers of suicide attempts
    main_plot(
        df_plot_texts_source_sex_age,
        df_plot_texts_source_all_pop,
        filename="residuals/eFigure2",  # residuals_age_sex
        conf_name=conf_name,
        y_col="residuals",
        ylabel="Model Residuals",
        add_model_estimated=False,
        set_ylim=False,
        main_legend="Model residuals",
    )

    # eFigure 3: Modality SA
    plot_modality_sa(
        dict_modality_sa, filename="modality/eFigure3_modality_sa", conf_name=conf_name
    )

    # eFigure 15: Kaplan-Meier curves relative to stay duration
    plot_kaplan_meier(
        **km_data,
        label_1=f"Pre-pandemic period (before {covid_date})",
        label_2=f"Post-pandemic period (after {covid_date})",
        label_1_short="Pre-pandemic",
        label_2_short="Post-pandemic",
        add_zoom=False,
        conf_name=conf_name,
        filename="severity/eFigure15_stay_duration",  # kaplan meier stay duration
    )

    # eFiugre 15bis Kaplan-Meier curves relative to death
    plot_kaplan_meier(
        **km_data_death,
        label_1=f"Pre-pandemic period (before {covid_date})",
        label_2=f"Post-pandemic period (after {covid_date})",
        label_1_short="Pre-pandemic",
        label_2_short="Post-pandemic",
        ylabel="Probability of still being alive",
        conf_name=conf_name,
        filename="severity/eFigure15bis_death",  # kaplan meier death
    )

    # eFigure 4: Monthly numbers of hospitalizations caused by suicide attempts - rule-based algorithm
    main_plot(
        df_plot_texts_source_sex_age_rule_based,
        df_plot_texts_source_all_pop_rule_based,
        filename="sensitivity_analysis/rule_based/eFigure4",  # rule_based
        conf_name=conf_name,
    )

    # eFigure 5: Monthly numbers of hospitalizations caused by suicide attempts - claim based algorithm
    main_plot(
        df_plot_claim_source_sex_age,
        df_plot_claim_source_all_pop,
        filename="sensitivity_analysis/claim/eFigure5",  # main_claim_data
        conf_name=conf_name,
    )

    # eFigure 6: Monthly numbers of hospitalizations caused by suicide attempts - completeness-adjusted
    main_plot(
        df_plot_texts_source_sex_age_rescaled,
        df_plot_texts_source_all_pop_rescaled,
        filename="sensitivity_analysis/rescaled/eFigure6",  # rescaled
        conf_name=conf_name,
        y_col="weighted_n_sa",
        main_legend="Hospitalised suicide attempts (rescaled)",
    )

    # eFigure7: Per-hospital forest plot of trend variations, overall population
    forest_plot(
        df=trend_dif_texts_source_all_pop_hosp.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/eFigure7",  # trend_dif_hospital
        conf_name=conf_name,
    )

    # eFigure8: Per-hospital forest plot of trend variations, young females
    forest_plot(
        df=trend_dif_texts_source_w_8_17_hosp.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/eFigure8",  # trend_dif_hospital_w_8_17
        conf_name=conf_name,
    )

    # eFigure9: Per-hospital forest plot of trend variations, overall population, rule-based algorithm
    forest_plot(
        df=trend_dif_texts_source_all_pop_hosp_rule_based.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/rule_based/eFigure9",  # trend_dif_hospital_rule_based
        conf_name=conf_name,
    )

    # eFigure10: Per-hospital forest plot of trend variations, young females, rule-based algorithm
    forest_plot(
        df=trend_dif_texts_source_w_8_17_hosp_rule_based.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/rule_based/eFigure10",  # trend_dif_hospital_rule_based_w_8_17
        conf_name=conf_name,
    )

    # eFigure11: Per-hospital forest plot of trend variations, overall population, completeness-adjusted
    forest_plot(
        df=trend_dif_texts_source_all_pop_hosp_rescaled.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/rescaled/eFigure11",  # trend_dif_hospital_rescaled
        conf_name=conf_name,
    )

    # eFigure12: Per-hospital forest plot of trend variations, young females, completeness-adjusted
    forest_plot(
        df=trend_dif_texts_source_w_8_17_hosp_rescaled.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/rescaled/eFigure12",  # trend_dif_hospital_w_8_17_rescaled
        conf_name=conf_name,
    )

    # eFigure13: Per-hospital forest plot of trend variations, overall population, claim-based algorithm
    forest_plot(
        df=trend_dif_claim_source_all_pop_hosp.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/claim/eFigure13",  # trend_dif_hospital_claim_data
        conf_name=conf_name,
    )

    # eFigure14: Per-hospital forest plot of trend variations, young females, claim-based algorithm
    forest_plot(
        df=trend_dif_claim_source_w_8_17_hosp.sort_values(
            ["label", "hospital_label"], ascending=[False, True]
        ),
        filename="sensitivity_analysis/claim/eFigure14",  # trend_dif_hospital_claim_data_w_8_17
        conf_name=conf_name,
    )


if __name__ == "__main__":
    """
    Parameters
    ----------
    conf_name: name of configuration file
    """
    typer.run(make_plots)
