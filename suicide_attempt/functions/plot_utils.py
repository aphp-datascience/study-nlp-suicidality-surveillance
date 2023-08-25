import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import StatisticalResult, logrank_test
from matplotlib import ticker
from pandas import DataFrame

from suicide_attempt.functions.constants import covid_date, sex_label
from suicide_attempt.functions.data_wrangling import filter_df_by_source_sex_age
from suicide_attempt.functions.utils import build_path

sns.set(rc={"figure.figsize": (11.7, 8.27)}, style="ticks")


def _save_plot(
    fig,
    filename,
    conf_name,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
    formats=["pdf", "png", "jpeg"],
):
    """
    Auxiliary function to save plot.
    """
    folder = os.path.expanduser(f"~/cse_210013/figures/results/{conf_name}")
    filenameimgs = [filename + "_" + conf_name + "." + format for format in formats]

    path_dir = build_path(__file__, folder)
    path_files = [os.path.join(path_dir, filenameimg) for filenameimg in filenameimgs]

    path_dir_extended = os.path.dirname(path_files[0])
    if not os.path.isdir(path_dir_extended):
        os.makedirs(path_dir_extended)

    for path_file, format in zip(path_files, formats):
        if legend is not None:
            fig.savefig(
                path_file,
                bbox_extra_artists=tuple(legend),
                bbox_inches="tight",
                format=format,
                dpi=300,
            )
        else:
            fig.savefig(path_file, format=format)

        print("Saved at:", path_file)

    plt.cla()
    plt.close("all")

    if tables:
        for i, table in enumerate(tables, start=1):
            filenametable = filename + "_" + str(i) + "_" + conf_name + ".csv"
            path_table = os.path.join(path_dir, filenametable)
            table.to_csv(path_table, index=False)

    print("Done -", filename)


def _show_or_save(
    fig,
    filename=None,
    conf_name=None,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
):
    # Show or save plot
    if (conf_name is not None) & (filename is not None):
        _save_plot(
            fig, filename=filename, conf_name=conf_name, legend=legend, tables=tables
        )

    else:
        plt.show()
        plt.cla()
        plt.close("all")


def plot_timeline(
    df: DataFrame,
    axes: List[plt.subplots],
    i: Optional[int],
    _col_: str = "n_sa",
    title: str = "",
    ylabel: str = "Monthly No. SA",
    add_covid_date_line: bool = False,
    percentage_col: bool = True,
    add_model_estimated: bool = True,
    model_estimate_col: str = "estimated_n_sa",
    add_error_bars: bool = True,
    set_ylim: bool = True,
    kwargs: Dict[str, Any] = {},
):
    """

    Parameters
    ----------
    df : DataFrame
    axes : List[plt.subplots]
    i : int
    _col_ : str, optional
        , by default "n_sa"
    title : str, optional
        , by default ""
    ylabel : str, optional
        , by default "Monthly No. SA"
    add_covid_date_line : bool, optional
        , by default False
    percentage_col : bool, optional
        , by default True
    add_model_estimated : bool, optional
        , by default True
    model_estimate_col : str, optional
        , by default "estimated_n_sa"
    add_error_bars : bool, optional
        , by default True
    set_ylim : bool, optional
        , by default True
    kwargs : Dict[str, Any], optional
        , by default {}
    """

    df = df.copy()

    # Define figure
    if i is not None:
        ax = axes[i]
    else:
        ax = axes

    # Create errorbars around points
    if add_error_bars:
        yerr = df[["std_residuals", "std_residuals"]].to_numpy()
        yerr = yerr.T
        ax.errorbar(
            x=df.date,
            y=df[model_estimate_col],
            yerr=yerr,
            fmt="none",
            capsize=5.2,
            color="grey",
            elinewidth=0.9,
        )

    # Axis 1
    sns.lineplot(data=df, ax=ax, x="date", y=_col_, dashes=False, **kwargs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    # add covid date line
    if add_covid_date_line:
        covid_date_dt = datetime.strptime(covid_date, "%Y-%m-%d")
        ax.axvline(covid_date_dt, ls=":", color="red")

    if add_model_estimated:
        sns.lineplot(
            ax=ax,
            data=df.query(f"date < '{covid_date}'"),
            x="date",
            y=model_estimate_col,
            color="grey",
            marker=None,
            legend=None,
            dashes=True,
            linestyle="--",
        )
        sns.lineplot(
            ax=ax,
            data=df.query(f"date >= '{covid_date}'"),
            x="date",
            y=model_estimate_col,
            color="grey",
            marker=None,
            legend=None,
            dashes=True,
            linestyle="--",
        )

    # Set x-ticks
    ax.set_xlabel("Date", fontsize=12)
    if set_ylim:
        ax.set_ylim(bottom=0)

    months_loc = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    ax.xaxis.set_major_locator(months_loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # assign locator and formatter for the xaxis ticks.
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")

    if percentage_col:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Set title of sub-plot
    ax.set_title(title, fontsize=14)

    # Grid
    ax.grid(visible=True, which="both", axis="both")


def main_plot(
    df_plot: DataFrame,
    df_plot2: DataFrame,
    title: str = None,
    ylabel: str = "Monthly No. hospitalised\n suicide attempts",
    y_col: str = "n_sa",
    h: int = 15,
    add_model_estimated: bool = True,
    model_estimate_col: str = "linear_trend",
    set_ylim: bool = True,
    percentage_col: bool = False,
    main_legend: str = "Hospitalised suicide attempts",
    sharey: bool = False,
    filename: str = None,
    conf_name: str = None,
    kwargs: Dict[str, Any] = dict(color="black", marker="o"),
):
    """Function to make the main plot of the article

    Parameters
    ----------
    df_plot : DataFrame
        df by sex and age
    df_plot2 : DataFrame
        df global (both sex and all ages)
    title : str
        title of the plot, optional
    filename : str, optional
        name of the file to save, if None so the plot will be print but not saved,
        by default None
    conf_name : str, optional
        name of configuration file, used for save,
        if None so the plot will be print but not saved, by default None
    """
    sns.set(style="ticks")
    # Age categories
    age_cats = df_plot.age_cat.cat.categories

    # Year categories
    years = df_plot.year.unique()
    years.sort()

    sex_cats = ["W", "M"]

    graphic_length = len(age_cats) + 1

    fig = plt.figure(
        figsize=(h, np.sqrt(2) * h),
    )
    gs = gridspec.GridSpec(nrows=graphic_length, ncols=4, wspace=0.2, hspace=0.8)

    ax0 = fig.add_subplot(gs[0, 1:3])

    plot_timeline(
        df_plot2,
        axes=ax0,
        i=None,
        title="Overall population",
        ylabel=ylabel,
        _col_=y_col,
        percentage_col=percentage_col,
        add_covid_date_line=True,
        kwargs=kwargs,
        add_error_bars=False,
        model_estimate_col=model_estimate_col,
        add_model_estimated=add_model_estimated,
        set_ylim=set_ylim,
    )

    axes = [[]]
    for i, age in enumerate(age_cats, start=1):
        for j, sex in enumerate(sex_cats, start=0):
            _j = j * 2

            if j >= 1 and sharey:
                ax = fig.add_subplot(gs[i, _j : _j + 2], sharey=axes[i - 1][0])

            else:
                ax = fig.add_subplot(gs[i, _j : _j + 2])

            # Filter data by age and sex
            df_f = filter_df_by_source_sex_age(df_plot, age, sex, None)

            title_sub = f"{sex_label[sex]} - Age: {age}"

            if j == 1:
                _ylabel = " "
            else:
                _ylabel = ylabel

            plot_timeline(
                df_f,
                axes=ax,
                i=None,
                title=title_sub,
                ylabel=_ylabel,
                _col_=y_col,
                percentage_col=percentage_col,
                add_covid_date_line=True,
                kwargs=kwargs,
                add_error_bars=False,
                model_estimate_col=model_estimate_col,
                add_model_estimated=add_model_estimated,
                set_ylim=set_ylim,
            )

            if sharey:
                if (j >= 1) | (i == 1):
                    axes[i - 1].append(ax)
                else:
                    axes.append([ax])

    # Aesthetics ###
    # Make figure legends
    black_line = mlines.Line2D([], [], color="red", linestyle=":", markersize=15)
    grey_line = mlines.Line2D([], [], color="grey", linestyle="--", markersize=15)
    main_line = mlines.Line2D([], [], color="black", linestyle="solid", markersize=15)

    labels_model = {
        "linear_trend": "Model linear component",
        "estimated_n_sa": "Model",
    }

    legend = fig.legend(
        loc=1,
        ncol=1,
        bbox_to_anchor=(0.95, 0.9),
        handles=[black_line, grey_line, main_line]
        if add_model_estimated
        else [black_line, main_line],
        labels=[
            "COVID-19 outbreak",
            labels_model[model_estimate_col],
            main_legend,
        ]
        if add_model_estimated
        else [
            "COVID-19 outbreak",
            main_legend,
        ],
        fontsize="small",
    )

    # Add Main title to figure
    if title is not None:
        plt.suptitle(title, fontsize=20, y=1)

    # Make tight layour
    cols2 = [
        "date",
        y_col,
    ]

    cols1 = ["date", y_col, "age_cat", "sex_cd", "year"]

    if add_model_estimated:
        cols1.append(model_estimate_col)
        cols2.append(model_estimate_col)

    # Show or save plot
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[
            df_plot2[cols2],
            df_plot[cols1],
        ],
    )


def p_value_getter(results: StatisticalResult, p_min=1e-3) -> str:

    p_value = results.p_value

    if p_value < p_min:
        return "p<1e-3"
    else:
        return f"p={p_value:.3f}"


def plot_kaplan_meier(
    T_1,
    E_1,
    T_2,
    E_2,
    label_1="Population A",
    label_1_short="Pop. A",
    label_2="Population B",
    label_2_short="Pop. B",
    title="",
    ylabel="Probability of still being hospitalised",
    xticks=list(range(0, 30, 1)),
    yticks=list(np.arange(0, 1.1, 0.1)),
    t_max=31,
    y_int=[0.0, 1.0],
    add_zoom: bool = True,
    conf_name=None,
    filename=None,
):
    """
    Generate a Kaplan Meirer plot of two populations

    Parameters
    ----------
    T_1 : List[np.array]
        duration subject was observed for in population A
    E_1 : List[boolean]
        True if the event was observed, False if the event was lost for population A
    T_2 : List[np.array]
        duration subject was observed for in population B
    E_2 : List[boolean]
        True if the event was observed, False if the event was lost for population B
    label_1 : str, delault = 'Population A'
        Label of the first population
    label_2 : str, delault = 'Population B'
        Label of the second population
    title : str, default = ''
        Title of the plot
    x_tickes : List, default = list(range(0,365,30))
        Range for the x-axis on the plot
    y_tickes : List(int), default = list(np.arange(0,1.1, 0.1))
        Range for the y-axis on the plot

    Returns
    -------
    fig, ax : matplotlib
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plt.title(title)
    plt.yticks(yticks)
    plt.xlim([0, t_max])
    plt.ylabel(ylabel, fontweight="bold")

    kmf = KaplanMeierFitter()
    kmf.fit(
        T_1, event_observed=E_1, timeline=np.array(range(t_max)), label=label_1_short
    ).plot_survival_function(ax=ax)

    kmf_2 = KaplanMeierFitter()
    kmf_2.fit(
        T_2, event_observed=E_2, timeline=np.array(range(t_max)), label=label_2_short
    ).plot_survival_function(ax=ax)

    loc = ticker.MultipleLocator(base=5.0)
    loc_minor = ticker.MultipleLocator(base=1.0)

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_minor_locator(loc_minor)
    ax.set_ylim(y_int)
    # Add at risk count
    add_at_risk_counts(
        kmf,
        kmf_2,
        ax=ax,
        rows_to_show=["At risk"],
        xticks=list(range(0, 31, 5)),
        ypos=-0.3,
    )
    ax.set_xlabel(
        xlabel="Duration after admission (days)",
        fontweight="bold",
    )
    ax.set_ylim(y_int)

    #####################
    if add_zoom:
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.35, 0.23, 0.5, 0.5]
        ax_zoom = fig.add_axes([left, bottom, width, height])

        kmf.plot_survival_function(ax=ax_zoom, label=label_1)
        kmf_2.plot_survival_function(ax=ax_zoom, label=label_2)
        handles, labels = ax_zoom.get_legend_handles_labels()
        ax_zoom.get_legend().remove()
        d = 0.015
        kwargs = dict(transform=ax_zoom.transAxes, color="k", clip_on=False)
        ax_zoom.plot((-d, d), (0.05 - d, 0.05 + d), **kwargs)
        ax_zoom.plot((-d, d), (0.03 - d, 0.03 + d), **kwargs)

        ax_zoom.xaxis.set_major_locator(loc)
        ax_zoom.xaxis.set_minor_locator(loc_minor)

        ax_zoom.set_xlim([0, t_max])

        ax_zoom.set_xlabel("")
    else:
        handles, _ = ax.get_legend_handles_labels()
        labels = [label_1, label_2]
    ############

    # LEGEND ###

    ax.legend().remove()
    legend = fig.legend(
        loc="upper right" if not add_zoom else "lower center",  # Position of legend
        ncol=1,  # Number of columns,
        bbox_to_anchor=(0.89, 0.85) if not add_zoom else (0.5, -0.25),
        handles=handles,
        labels=labels,
        fontsize="small",
    )

    print(f" {str(t_max)} days duration Rate : {kmf.survival_function_.iloc[-1]}\n")
    print(f" {str(t_max)} days duration Rate : {kmf_2.survival_function_.iloc[-1]}\n")

    print("Log Rank Test : \n")
    results = logrank_test(T_2, T_1, event_observed_A=E_2, event_observed_B=E_1)
    # Petit p
    ax.text(0.2, 0.87, f"    {p_value_getter(results)} by log-rank test")

    print(results.print_summary())
    print(f"La p-value est de {results.p_value}")

    # Show or save plot
    datadict = {"T_1": T_1, "T_2": T_2, "E_1": E_1, "E_2": E_2}
    data = pd.DataFrame(datadict)
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[data],
    )


def forest_plot(
    df: DataFrame,
    title: str = None,
    xcol: str = "delta_t",
    ycol: str = "hospital_label",
    ylabel: str = "Hospital",
    xlabel: str = "Trend variation after the COVID-19 outbreak of\n the monthly number of hospitalised suicide attempts",  # noqa: E501
    aesthetic_args: Dict[str, Any] = dict(
        color="black",
        hue="label",
        style="label",
        size="label",
        sizes={"all_pop": 100, "by_hosp": 400},
        markers={"by_hosp": ".", "all_pop": "D"},
    ),
    cols_labels=["Hospital", "Hospital type", "No. hospitalised\nsuicide attempts"],
    cols_text=["hospital_label", "hospital_type", "n_stays"],
    filename: str = None,
    conf_name: str = None,
):

    sns.set(rc={"figure.figsize": (11.7, 8.27)}, style="ticks")
    xerr = df[["lower_delta", "upper_delta"]].to_numpy()
    xerr = xerr.T

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create errorbars around points
    ax.errorbar(
        df[xcol],
        df[ycol],
        xerr=xerr,
        fmt="none",
        capsize=5.2,
        color="grey",
        elinewidth=0.9,
    )

    # Create points
    sns.scatterplot(x=xcol, y=ycol, data=df, ax=ax, legend=False, **aesthetic_args)

    # Create line at 0
    plt.plot([0, 0], ax.yaxis.get_data_interval(), "--", lw=2, color="grey")

    # Set labels
    ax.set(xlabel=xlabel, ylabel=None)
    ax.yaxis.set_ticklabels([])

    # Create table
    cell_text = df[cols_text].astype(str).values
    n = len(df[ycol])
    the_table = plt.table(
        cellText=cell_text,
        colLabels=cols_labels,
        cellLoc="center",
        bbox=(-0.6, 0.0, 0.6, (n + 1) / n),
        colWidths=[0.2, 0.4, 0.33],
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)

    # Grid
    ax.grid(visible=True, which="both", axis="both")

    # Set title
    if title is not None:
        plt.suptitle(title, fontsize=20, y=1)

    # Set legend parameters
    fig.tight_layout()

    # Show or save plot
    cols = [xcol, ycol] + cols_text
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            the_table,
        ],
        tables=[df[cols]],
    )


def plot_rf(
    data: DataFrame,
    titles: Dict[str, str] = {
        "Sexual violence": "sexual_violence_ratio",
        "Domestic violence": "domestic_violence_ratio",
        "Physical violence": "physical_violence_ratio",
        "Social isolation": "social_isolation_ratio",
        "Suicide attempt history": "has_history_ratio",
    },
    hue_dict: Dict[str, str] = {
        "all_pop": {"label": "Overall population", "marker": ">"},
        "M": {"label": "Male", "marker": "v"},
        "W": {"label": "Female", "marker": "^"},
    },
    hue_key: str = "sex_cd",
    ylabel: str = "Risk Factor Prevalence",
    filename: str = None,
    conf_name: str = None,
):

    # Marker style
    filled_marker_style = dict(
        linestyle="-",
        markersize=8,
        markeredgecolor="black",
        markers={key: hue_dict[key]["marker"] for key in hue_dict.keys()},
    )

    # Define figure ##
    # Initialize figure
    sns.set(rc={"figure.figsize": (11.7, 8.27)}, style="ticks")

    # Initialize figure
    h = 14
    fig, axes = plt.subplots(
        nrows=len(titles),
        ncols=1,
        figsize=(h, np.sqrt(2) * h),  # (15,45)
        sharex=False,
        sharey=False,
    )

    for ax, title_tup in zip(axes, titles.items()):
        sns.lineplot(
            data=data,
            ax=ax,
            x="date",
            y=title_tup[1],
            hue=hue_key,
            dashes=False,
            legend="auto",
            hue_order=hue_dict.keys(),
            style=hue_key,
            **filled_marker_style,
        )

        # Set labels and aesthetics ##
        # Assign locator and formatter for the xaxis ticks. (for date axis)
        months_loc = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        ax.xaxis.set_major_locator(months_loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("right")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Grid
        ax.grid(visible=True, which="both", axis="both")

        covid_date_dt = datetime.strptime(covid_date, "%Y-%m-%d")
        ax.axvline(covid_date_dt, ls=":", color="red")

        ax.set_title(title_tup[0], fontsize=14)

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel("Date", fontsize=14)

        # Set legend
        handles, labels = ax.get_legend_handles_labels()
        hue_labels = [hue_dict[key]["label"] for key in labels]
        legend = ax.legend(
            handles,
            hue_labels,
            loc="upper left",
            ncol=1,
            bbox_to_anchor=(0.01, 0.97),
        )

    fig.tight_layout()

    # Show or save plot
    cols = ["date", hue_key] + list(titles.values())
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[data[cols]],
    )


def plot_ratio_docs_visit(df: DataFrame, filename: str = None, conf_name: str = None):
    """Function to plot the Rho ratio:
    Proportion of stays w\ at least one discharge summary

    Parameters
    ----------
    df : DataFrame
        should have at least the following columns: `date`, `rho`, `hospital_label`
    filename : str, optional
        , by default None
    conf_name : str, optional
        , by default None
    """

    labels = df.hospital_label.unique()

    ylabel = "Proportion of stays w\ at least one discharge summary"
    sns.set(rc={"figure.figsize": (11.7, 8.27)}, style="ticks")
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        ax=ax,
        x="date",
        y="rho",
        dashes=True,
        hue="hospital_label",
        hue_order=labels,
        style="hospital_label",
        legend=False,
        linewidth=2.8,
    )
    ax.set_ylim(bottom=0)
    months_loc = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    ax.xaxis.set_major_locator(months_loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # assign locator and formatter for the xaxis ticks.
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(visible=True, which="both", axis="both")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Date", fontsize=14)

    # Make figure legends
    legend = fig.legend(
        labels=labels,
        loc="lower right",
        borderaxespad=0.8,
        title="Hospital trigram",
        ncol=3,
        bbox_to_anchor=(0.99, 0.15),
        fontsize=14,
    )
    fig.tight_layout()

    # Show or save plot
    cols = ["date", "rho", "hospital_label"]
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[df[cols]],
    )


def plot_modality_sa(
    data_dict,
    filename: str = None,
    conf_name: str = None,
    ylabel: str = "Proportion of each suicide attempt modality",
):
    # sns.set_theme()

    palette = sns.palettes.SEABORN_PALETTES["muted"]

    palette = [
        "313335",
        "e68523",
        "4285f4",
        "34a853",
        "ea4335",
        "fbbc05",
    ]

    fig, axes = plt.subplots(
        figsize=(16, 20),
        nrows=len(data_dict),
        ncols=1,
    )
    # Set layout
    fig.tight_layout()

    # set the spacing between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )

    # Month locator
    months_loc = mdates.MonthLocator(bymonth=[1, 4, 7, 10])

    # Plot
    for ax, data_item in zip(axes, data_dict.items()):
        title = data_item[0]
        data = data_item[1]

        # Extract labels
        cols = [i for i in data.columns if i not in ["date", "sex_cd", "age_cat"]]

        ax.set_title(title, fontdict={"fontsize": 16})
        stacks = ax.stackplot(
            data.date,
            data.drop(columns=["date", "sex_cd"], errors="ignore").T,
            labels=list(cols),
            colors=palette,
        )

        hatches = ["//", "\\\\", "||", "--", "++", "xx", "oo", "OO", "..", "**"]
        for stack, hatch in zip(stacks, hatches[: len(stacks)]):
            stack.set_hatch(hatch)

        # Assign locator and formatter for the xaxis ticks. (for date axis)
        ax.xaxis.set_major_locator(months_loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("right")
            label.set_fontsize(14)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Label
        ax.set_ylabel(ylabel, fontsize=16)

        # Grid
        ax.grid(visible=True, which="both", axis="both")
        ax.legend().remove()

        # add covid date line
        covid_date_dt = datetime.strptime(covid_date, "%Y-%m-%d")
        ax.axvline(covid_date_dt, ls="--", color="black")

    legend = plt.legend(
        loc="upper right",
        borderaxespad=0.8,
        ncol=2,
        bbox_to_anchor=(0.95, -0.325),
        fontsize=14,
    )

    # Show or save plot
    _show_or_save(
        fig,
        filename=filename,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=list(data_dict.values()),
    )
