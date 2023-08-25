import pandas as pd
import typer

from suicide_attempt.functions.data_wrangling import transform_rf_stays_to_long
from suicide_attempt.functions.stats_utils import GetSAStaysNLP, ValidationMetrics
from suicide_attempt.functions.utils import save_file


def validation_and_age_description(conf_name):

    SANLP = GetSAStaysNLP(conf_name)

    df = SANLP.get_nlp_stays()
    # # Data description (basics)

    data_description = df.query("~recurrent_visit_ml & ml_positive").agg(
        age_mean=("age", "mean"), age_std=("age", "std")
    )

    # # SA-RB
    # ## (a)
    config = dict(
        annotation_subset="SA-RB",
        annotators=["benjamin", "vincent"],
        label_y_pred="rb_positive",
        data_to_validate=df.query("rb_positive & ~ml_positive"),
    )

    vm = ValidationMetrics(**config)
    ppv_rb_only, _, _ = vm.precision_by_period()

    # ## (b)
    config = dict(
        annotation_subset="SA-ML",
        annotators=["benjamin", "vincent"],
        label_y_pred="rb_positive",
        data_to_validate=df.query("rb_positive & ml_positive"),
    )

    vm = ValidationMetrics(**config)
    ppv_ml_rb, _, _ = vm.precision_by_period()

    # ## (c)
    p_ml_given_rb = (
        df.query("rb_positive").ml_positive.value_counts(normalize=True).loc[True]
    )

    # ## (d)
    ppv_rb = p_ml_given_rb * ppv_ml_rb + (1 - p_ml_given_rb) * ppv_rb_only

    # # SA-ML
    config = dict(
        annotation_subset="SA-ML",
        annotators=["benjamin", "vincent"],
        label_y_pred="ml_positive",
        data_to_validate=df,
    )

    vm = ValidationMetrics(**config)
    sa_ml_summary = vm.summary()

    print(vm.compute_agreement())

    # # RF
    df_rf = transform_rf_stays_to_long(df)

    config = dict(
        annotation_subset="RF",
        annotators=["benjamin", "vincent"],
        label_y_pred="rf_prediction",
        data_to_validate=df_rf.query("rf_prediction"),
    )

    vm = ValidationMetrics(**config)

    # + tags=[]
    rf_summary = vm.summary()
    # -

    print(vm.compute_agreement())

    # Table 2
    summary = rf_summary
    summary["sa_ml"] = sa_ml_summary
    summary["sa_rb"] = {
        vm.periods[period]: f"{ppv_rb.loc[period]:.2f}" for period in ppv_rb.index
    }

    table2 = pd.DataFrame(summary).T

    old_index = table2.index
    new_index = pd.Categorical(
        old_index,
        categories=[
            "sa_ml",
            "sa_rb",
            "social_isolation",
            "domestic_violence",
            "sexual_violence",
            "physical_violence",
            "has_history",
        ],
        ordered=True,
    )

    table2.index = new_index

    table2.sort_index(inplace=True)

    return table2, data_description


def main(conf_name: str = typer.Argument(..., help="name of conf file")):
    # Execute pipeline
    table2, data_description = validation_and_age_description(conf_name=conf_name)

    # Save
    save_file(
        file=table2.to_csv(index=True),
        conf_name=conf_name,
        name="table2.csv",
    )
    save_file(
        file=data_description.to_csv(index=True),
        conf_name=conf_name,
        name="data_description.csv",
    )


if __name__ == "__main__":

    typer.run(main)
