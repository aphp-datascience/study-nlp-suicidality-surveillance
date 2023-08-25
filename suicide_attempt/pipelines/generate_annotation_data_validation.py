import os
from typing import List

import pandas as pd
import typer

from suicide_attempt.functions import utils
from suicide_attempt.functions.annotation_utils import ValidationDatasetCreation


def main(
    conf_name: str = typer.Argument(..., help="Name of the conf file"),
    annotation_subset: str = typer.Argument(
        ..., help="Type of data. One of {'SA-ML','SA-RB','RF'} "
    ),
    n1: int = typer.Argument(..., help="Sample size"),
    annotator_names: List[str] = typer.Option(
        ["a1", "a2", "a3"], help="List of annotator names"
    ),
    avoid_list: List[str] = typer.Option(None, help="List of encounter_num to avoid"),
    share_p: float = typer.Option(
        0.1, help="Number of stays that annotators have in common with another one"
    ),
    supplementary: bool = typer.Option(False, help="is supplementary data"),
):
    print(annotator_names)
    # Save results
    path = os.path.expanduser(
        f"~/cse_210013/data/annotation/validation/raw_validation/{conf_name}/{annotation_subset}/raw_{annotation_subset}_{conf_name}"  # noqa: E501
    )

    if (supplementary) and (len(avoid_list) == 0):
        existing_data = pd.read_pickle(path)
        avoid_list = list(existing_data.encounter_num.unique())

    vdc = ValidationDatasetCreation(
        conf_name=conf_name,
        annotation_subset=annotation_subset,
        annotator_names=annotator_names,
        n1=n1,
        avoid_list=avoid_list,
        share_p=share_p,
    )
    data = vdc.get_data()

    if supplementary:
        path += "_supplementary"
    _directory = utils.get_dir_path(path)
    if not os.path.isdir(_directory):
        os.makedirs(_directory)
    data.to_pickle(path)

    if not supplementary:
        vdc.generate_annotation_notebooks()


if __name__ == "__main__":

    typer.run(main)
