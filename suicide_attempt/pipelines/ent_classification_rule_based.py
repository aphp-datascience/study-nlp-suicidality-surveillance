import os
import time
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import typer
from edsnlp.processing import pipe as edsnlp_pipe

from suicide_attempt.functions import utils
from suicide_attempt.functions.constants import regex_rf, regex_sa
from suicide_attempt.functions.data_wrangling import (
    split_process_and_link,
    tag_history_w_date,
)
from suicide_attempt.functions.text_utils import nlp_baseline, pick_results


def ent_classification_rule_based(
    df: pd.DataFrame,
    conf_name: Optional[str],
    parallelize: bool = True,
    regex: Optional[Union[str, Dict[str, List[str]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline to identify lexical variant modifiers.
    (History, negation, family context, hypothesis and dates).
    It uses the NLP baseline functions from the edsnlp library.

    Parameters
    ----------
    df: pd.DataFrame, datafame with notes where to apply the pipeline.
    The df should have at least the columns
    `note_id`, `note_text` , `visit_start_date`, `birth_date`
    conf_name: str, configuration file name.
    It will be used to retrieve the data from stay_document_selection
    and the regex patterns.
    parallelize : bool,
        Whether to apply the Spacy pipeline in parallel mode. By default =True
    regex: Optional[Union[str, Dict[str, List[str]]]], default = None,
        Regex to match


    Returns
    -------
    ents: pd.DataFrame, the df contains one line by entity.
        The columns of the df are
        ['encounter_num','patient_num','note_id','lexical_variant','label',
        'start','start_char','end','end_char','snippet','negated','family',
        'history','hypothesis',]
    dates: pd.DataFrame, the df contains one line by date.
    """
    if conf_name:
        parameters = utils.get_conf(conf_name)

    # Import regex
    if (regex is None) and conf_name:
        # Import parameters
        regex = regex_sa
    elif isinstance(regex, dict):
        regex = regex
    elif regex == "rf":
        regex = regex_rf
    else:
        raise ValueError

    # Import the NLP baseline for negation, familly context, history and hypothesis
    nlp = nlp_baseline(regex=regex)

    # Apply the pipeline and retrieve results
    t3 = time.time()

    if parallelize:
        ents_df = edsnlp_pipe(
            note=df,
            nlp=nlp,
            results_extractor=pick_results,
            n_jobs=-2,
            progress_bar=True,
            context=["visit_start_date"],
        )
    else:
        ents_df = edsnlp_pipe(
            note=df,
            nlp=nlp,
            results_extractor=pick_results,
            n_jobs=1,
            progress_bar=True,
            context=["visit_start_date"],
        )

    # Merge ents and dates to df
    df.drop(columns=["note_text"], inplace=True)
    df = df.merge(ents_df, on="note_id", how="inner")

    t4 = time.time()

    print(f"Spacy Pipeline applied in {(t4-t3)/60:.3f} min")

    # Split, process and link dates with entities
    ents, dates = split_process_and_link(df, exclude_birthdate=False)

    # Tag history if delta is greater than constant
    ents = tag_history_w_date(
        ents, threshold=parameters["delta_history"], exclude_birthdate=True
    )

    return ents, dates


def main(
    conf_name: str = typer.Argument(..., help="Name of the conf file"),
    parallelize: bool = typer.Option(
        True, help="Whether to apply the Spacy pipeline in parallel mode."
    ),
    use_cached_data: bool = typer.Option(
        False, help="Whether to use a sample of cached data. For testing purposes."
    ),
    debug: bool = typer.Option(
        False, help="Whether to use a subsample data. For testing purposes."
    ),
    regex: Optional[str] = typer.Option(None, help="regex to use"),
    file_name_in: str = typer.Option(
        "stay_document_selection_", help="File name to read input data"
    ),
    file_name_out: str = typer.Option(
        "result_ent_classification_rule_based_", help="File name to save results"
    ),
):
    """
    If name=='__main__':
    the entire pipeline will be run with the data of the specified
    configuration file (or cached data).
    The results will be saved by default at
    'cse_210013/data/{conf_name}/result_ent_classification_rule_based_{conf_name}'

    Parameters
    ----------
    conf_name : str,
        Name of the conf file
    parallelize : bool,
        Whether to apply the Spacy pipeline in parallel mode. By default=True
    use_cached_data : bool
        Whether to use a sample of cached data. For testing purposes. By default=False
    debug : bool
        Whether to use a subsample data. For testing purposes. By default=False
    regex: str, default=None
        regex to use. If None, it will use the regex of the configuration file
    file_name_in: str, default = 'stay_document_selection_'
        File name to read input data
    file_name_out: str, default = 'result_ent_classification_rule_based_'
        File name to save results
    """
    # Import data
    t1 = time.time()
    if use_cached_data:
        df = pd.read_pickle(file_name_in)

    else:
        df = utils.read_parquet_from_hdfs(file_name=f"{file_name_in}{conf_name}")
        if debug:
            df = df.sample(100, random_state=0)

    t2 = time.time()
    print(f"Data loaded in {t2-t1:.3f}s")
    print("Dataset length:", len(df))

    # Execute pipeline
    ents, _ = ent_classification_rule_based(
        df=df, conf_name=conf_name, parallelize=parallelize, regex=regex
    )

    # Export Results
    save_dir_path = os.path.join(os.path.expanduser("~/cse_210013/data"), conf_name)

    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path)

    file_path = os.path.abspath(
        os.path.join(save_dir_path, f"{file_name_out}{conf_name}")
    )

    ents.to_pickle(file_path)
    print("Results saved at:", file_path)


if __name__ == "__main__":
    typer.run(main)
