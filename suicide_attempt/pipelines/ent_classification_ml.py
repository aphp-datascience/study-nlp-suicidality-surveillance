import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import typer
from torch.utils.data import DataLoader

from suicide_attempt.functions.nlp_ml.bert_models import BertTokenClassification
from suicide_attempt.functions.nlp_ml.ml_utils import PLDataset, collate_fn_inference

print(torch.__version__)


def ent_classification_ml(
    data_path: Optional[Path] = typer.Option(None),
    conf_name: Optional[str] = typer.Option(None),
    path_results: Optional[Path] = typer.Option(None),
    path_model: Path = typer.Option("~/cse_210013/data/models/BertTokenEDS.ckpt"),
    device: str = typer.Option("cuda"),
):
    """Pipeline to make inference with CamemBERT for Token Classification trained
     to classify SA entities as positive or negative mentions.

    Parameters
    ----------
    data_path : Optional[Path], default=None
        path to a data related to home
        (ex: '~/cse_210013/data/confXX/result_ent_classification_rule_based_confXX' ).
        It should be a pandas.DataFrame pickle.
        The df should have the columns `words` and `word_index`.
    conf_name : Optional[str], default=None.
        Either data_path or conf_name should be given.
    path_results : Optional[Path]
        path to where to save the results, by default None.
        If None,
        path_results = (
            f"~/cse_210013/data/{conf_name}/result_ent_classification_ml_{conf_name}"
        )
        The df has the following columns :
         ["note_id", "start_char", "end_char", "bert_token_prediction"]
    path_model : Path, optional
        path to CamemBERT for Token Classification Model,
         by default "~/cse_210013/data/models/BertTokenEDS.ckpt"
    device : str, optional
        {'cpu','cuda'}, by default "cuda"
    """
    # Read data
    if (data_path is None) and (conf_name is None):
        raise ValueError("`data` or `conf_name` should be not null")

    if data_path:
        data_path = os.path.expanduser(data_path)
    elif conf_name:
        data_path = f"~/cse_210013/data/{conf_name}/result_ent_classification_rule_based_{conf_name}"  # noqa: E501

    df = pd.read_pickle(data_path)

    # Check if df has an unique index
    assert df.index.is_unique, "data df index should be unique"

    # Cast as type int
    df["word_index"] = df["word_index"].astype(int)

    path_model = os.path.expanduser(path_model)

    # Load model
    model = BertTokenClassification.load_from_checkpoint(path_model)

    if device == "cuda":
        model = model.to(torch.device("cuda"))
    print("model.device.type:", model.device.type)

    # Initialize the trainer
    trainer = pl.Trainer(gpus=0, logger=False)

    # Validate (just to be sure that it is the expected model)
    trainer.validate(model)[0]

    # Create a torch dataset from pandas
    inference_ds = PLDataset(df)

    # Create a dataloader
    inference_dl = DataLoader(
        inference_ds,
        batch_size=32,
        num_workers=2,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_inference,
    )

    # Initialize the trainer object and predict
    if device == "cuda":
        trainer = pl.Trainer(gpus=1, accelerator="dp", logger=False)
    else:
        trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=inference_dl)

    # Concatenate all
    predictions_cat = torch.cat(predictions).cpu().numpy()

    # Assign as a pandas column
    df["bert_token_prediction"] = predictions_cat

    # Save
    if path_results is None:
        path_results = (
            f"~/cse_210013/data/{conf_name}/result_ent_classification_ml_{conf_name}"
        )

    df[["note_id", "start_char", "end_char", "bert_token_prediction"]].to_pickle(
        path_results
    )


if __name__ == "__main__":

    typer.run(ent_classification_ml)
