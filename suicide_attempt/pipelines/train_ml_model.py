import os
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torchmetrics
import typer
from pytorch_lightning.loggers import TensorBoardLogger

from suicide_attempt.functions.nlp_ml.bert_models import BertTokenClassification

print("torch", torch.__version__)
print("pl", pl.__version__)
print("torchmetrics", torchmetrics.__version__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_bert_token(
    camembert_model_path,
    name,
    max_epochs,
    path_dataset,
    lr,
    batch_size,
    save_top_k,
    n_gpus,
):
    bert = BertTokenClassification(
        camembert_model_path=camembert_model_path,
        config_optimizer=dict(lr=lr, weight_decay=0.1, betas=(0.9, 0.98), eps=1e-6),
        finetune=True,
        batch_size=batch_size,
        dropout=0.1,
        class_weights=None,
        num_epoch=max_epochs,
        num_warmup_epoch=2,
        scheduler=dict(scheduler="LambdaLR", verbose=True),
        data_dir=path_dataset,
    )

    # Initialize the logger
    logger = TensorBoardLogger(
        save_dir="/export/home/cse210013/tensorboard_data",
        name=name,
        default_hp_metric=False,
    )

    # Early stopping
    monitor = "valid/f1"
    early_stopping = pl.callbacks.EarlyStopping(monitor, patience=7, mode="max")

    # Checkpoint
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="valid/f1",
        mode="max",
        save_top_k=save_top_k,
        every_n_epochs=1,
        dirpath="/export/home/cse210013/cse_210013/data/models",
        filename=name,
    )

    # Lr monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # Initialize the trainer
    trainer = pl.Trainer(
        gpus=n_gpus,
        auto_select_gpus=True,
        accelerator="dp",
        max_epochs=max_epochs,
        callbacks=[
            checkpoint,
            lr_monitor,
            early_stopping,
        ],
        logger=logger,
        log_every_n_steps=10,
        # profiler="advanced"
    )

    # Fit the model
    trainer.fit(bert)

    return trainer


def main(
    path_dataset: Path = typer.Option(
        "~/cse_210013/data/datasets/annotation_results_clean_k1=35_k2=10",  # noqa: E501
        help="Path to the dataset",
    ),
    camembert_model_path: Path = typer.Option(
        "~/word-embedding/training-from-scratch-2021-08-13",
        help="Path to the camembert language model",
    ),
    max_epochs: int = typer.Option(10, help="Max number of epochs"),
    name: str = typer.Option("hp_search_BertToken_eds", help="name to save"),
    hp_search: bool = typer.Option(
        True, help="whether to make a grid search or train a single model"
    ),
    lr_list: List[float] = typer.Option(
        [1e-5, 2e-5, 3e-5, 5e-5], help="learning rate list for hp search. "
    ),
    batch_size_list: List[int] = typer.Option(
        [16, 32], help="batch size lsit for hp search."
    ),
    batch_size: int = typer.Option(32, help="batch size. Recommended value 32"),
    lr: float = typer.Option(3e-5, help="Learning rate. Recommended value 3e-5"),
    n_gpus: int = typer.Option(2, help="number of gpus"),
):
    # Parameters
    path_dataset = os.path.expanduser(path_dataset)
    camembert_model_path = os.path.expanduser(camembert_model_path)

    if hp_search:

        for _lr in lr_list:
            for _batch_size in batch_size_list:
                print("lr", _lr)
                print("batch_size", _batch_size)
                _ = train_bert_token(
                    camembert_model_path=camembert_model_path,
                    max_epochs=max_epochs,
                    path_dataset=path_dataset,
                    name=name,
                    lr=_lr,
                    batch_size=_batch_size,
                    save_top_k=0,
                    n_gpus=n_gpus,
                )
    else:
        assert isinstance(batch_size, int)
        assert isinstance(lr, float)

        trainer = train_bert_token(
            camembert_model_path=camembert_model_path,
            max_epochs=max_epochs,
            path_dataset=path_dataset,
            name=name,
            lr=lr,
            batch_size=batch_size,
            save_top_k=1,
            n_gpus=n_gpus,
        )

        trainer.validate(ckpt_path="best")[0]


if __name__ == "__main__":

    typer.run(main)
