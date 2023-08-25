import os
from itertools import chain
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CamembertModel, PretrainedConfig

from suicide_attempt.functions.nlp_ml.metrics import Metrics
from suicide_attempt.functions.nlp_ml.ml_utils import (
    PLDataset,
    tokenize_and_align_labels,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, dropout, hidden_size, num_classes):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertTokenBase(pl.LightningModule):
    """
    Custom Pytorch Lightning module to train text classification models
    using a transformer layer followed by a simple classifier.
    WARNING: it is currently available only for binary classification.

    Parameters
    ----------
    bert_model: str
        Path of the CamembertModel to load.
    num_classes: int
        Number of label classes.
    finetune: bool
        Whether to finetune (i.e modify the last layers) the BERT model or not.
    dropout: float
        Dropout probability for the classifier.
    lr: float
        Learning rate to reach after the warmup steps and before the decay.
    class_weights: Optional[Dict[str, float]]
        Weights to apply to each class.
    num_epoch: int
        Number of training epochs (only useful for warmup)
    num_warmup_epoch: int
        Number of warmup epoch for the learning rate.
    scheduler: dict
        Parameters of scheduler. Default= dict(scheduler='ReduceLROnPlateau',
        factor=.5,patience=2,verbose=True)
    """

    def __init__(
        self,
        camembert_model_path: str,
        num_classes: int = 2,
        finetune: bool = True,
        dropout: float = 0.1,
        config_optimizer={"lr": 1e-4},
        class_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        num_epoch: int = 10,
        num_warmup_epoch: int = 1,
        scheduler: dict = dict(
            scheduler="ReduceLROnPlateau", factor=0.5, patience=2, verbose=True
        ),
        data_dir="../../data/datasets/annotation_results_conf14_clean_220114_k1=35_k2=10",  # noqa: E501
    ):
        super().__init__()

        self.camembert_model_path = camembert_model_path

        if self.camembert_model_path[-4:] == "json":
            config = PretrainedConfig()
            config = config.from_json_file(self.camembert_model_path)
            self.bert = CamembertModel(config)
        else:
            self.bert = CamembertModel.from_pretrained(self.camembert_model_path)

        self.finetune = finetune
        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.pooler.dense.out_features, num_classes),
        )

        if class_weights is not None:
            class_weights = torch.tensor(class_weights.values(), dtype=torch.float)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        self.train_metrics = Metrics()
        self.valid_metrics = Metrics()

        self.config_optimizer = config_optimizer
        self.lr = config_optimizer["lr"]
        self.num_epoch = num_epoch
        self.num_warmup_epoch = num_warmup_epoch

        self.scheduler = scheduler
        self.batch_size = batch_size
        self.data_dir = data_dir

        # Initialize a tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.camembert_model_path)

        # HP
        self.save_hyperparameters()

    def _apply_model(self, batch):
        """Apply bert + classification head.

        Output shape is (batch_size,sequence length, num_classes)

        Parameters
        ----------
        batch :

        Returns
        -------
        x
        """
        x = self.bert(**batch).last_hidden_state
        y = self.classification_head(x)
        return y

    def forward(self, batch):

        tokenized_input = self.tokenizer(
            batch["words"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        if self.device.type == "cuda":
            for key, values in tokenized_input.items():
                tokenized_input[key] = values.to(self.device)

        x = self._apply_model(tokenized_input)
        predictions = x.argmax(-1)
        # predictions = predictions.cpu()#.numpy()

        if "word_index" in batch.keys():

            token_index_list = []
            for i, (word_index, t) in enumerate(zip(batch["word_index"], predictions)):
                token_index = tokenized_input.word_to_tokens(
                    batch_or_word_index=i, word_index=word_index
                ).start
                token_index_list.append(token_index)

            predictions = predictions[
                (
                    torch.arange(start=0, end=len(batch["word_index"])),
                    torch.tensor(token_index_list),
                )
            ]

        return predictions

    def _step(self, batch, labels):
        """Used in training and validation
        Compute loss and predictions

        Parameters
        ----------
        batch :
        labels :

        Returns
        -------
        Tuple(loss, predicitions)
        """
        x = self._apply_model(batch)
        loss = self.criterion(
            x.permute(0, 2, 1), labels
        )  # loss(logits.view(-1, 2), labels.view(-1))
        predictions = x.argmax(-1)

        return loss, predictions

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        labels = batch.pop("labels")

        loss, predictions = self._step(batch, labels)

        return {"loss": loss, "preds": predictions, "labels": labels}

    def training_step_end(self, outputs):
        # update and log
        metrics = self.train_metrics(
            preds=outputs["preds"],
            target=outputs["labels"],
        )

        outputs["loss"] = outputs["loss"].sum()

        self.log(
            "train/f1_step",
            metrics["f1"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/accuracy_step",
            metrics["accuracy"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/precision_step",
            metrics["precision"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        self.log(
            "train/loss_step",
            outputs["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
        )

        return outputs

    def training_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()

        train_loss = sum([output["loss"] for output in outputs])

        self.log(
            "train/f1",
            metrics["f1"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/accuracy",
            metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/precision",
            metrics["precision"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):

        labels = batch.pop("labels")

        self.eval()
        with torch.no_grad():
            loss, predictions = self._step(batch, labels)
        self.train()

        return {"loss": loss, "preds": predictions, "labels": labels}

    def validation_step_end(self, outputs):
        # update and log
        metrics = self.valid_metrics(outputs["preds"], outputs["labels"])

        outputs["loss"] = outputs["loss"].sum()

        self.log(
            "valid/f1_step",
            metrics["f1"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            "valid/accuracy_step",
            metrics["accuracy"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            "valid/precision_step",
            metrics["precision"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            "valid/loss_step",
            outputs["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        return outputs

    def validation_epoch_end(self, outputs):
        metrics = self.valid_metrics.compute()

        valid_loss = sum([output["loss"] for output in outputs])

        self.log("valid/f1", metrics["f1"])
        self.log("valid/accuracy", metrics["accuracy"])
        self.log("valid/precision", metrics["precision"])
        self.log("valid/loss", valid_loss)

        self.valid_metrics.reset()

    def configure_optimizers(self):
        if self.finetune:
            params = self.parameters()
        else:
            params = self.classification_head.parameters()
            if self.crf:
                params = chain(params, self.crf.parameters())

        optimizer = torch.optim.Adam(params, **self.config_optimizer)

        if self.scheduler["scheduler"] == "LambdaLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, self.lr_lambda, verbose=True
            )
        if self.scheduler["scheduler"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler["factor"],
                patience=self.scheduler["patience"],
                verbose=self.scheduler["verbose"],
            )

        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor="train/loss")

    def lr_lambda(self, current_epoch: int):
        if current_epoch <= self.num_warmup_epoch:
            return float(current_epoch) / float(max(1, self.num_warmup_epoch))
        else:
            lr_range = self.lr
            decay_steps = self.num_epoch - self.num_warmup_epoch
            pct_remaining = 1 - (current_epoch - self.num_warmup_epoch) / decay_steps
            decay = lr_range * pct_remaining
            return decay / self.lr  # as LambdaLR multiplies by lr_init

    def prepare_data(self):
        # Load data
        datasets = load_from_disk(self.data_dir)

        # Split into different objects for convenience
        train = datasets["train"]
        validation = datasets["validation"]

        # Create a dictionary for each set
        train_tokens = tokenize_and_align_labels(
            text=train["words"],
            tags=train["word_label"],
            tokenizer=self.tokenizer,
            max_length=512,
            label_all_tokens=False,
        )

        val_tokens = tokenize_and_align_labels(
            text=validation["words"],
            tags=validation["word_label"],
            tokenizer=self.tokenizer,
            max_length=512,
            label_all_tokens=False,
        )

        # Create datasets
        train_dataset = PLDataset(train_tokens)
        validation_dataset = PLDataset(val_tokens)

        # Dataset loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            drop_last=False,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validation_loader

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "valid/f1": 0,
                "valid/precision": 0,
            },
        )


class BertTokenClassification(BertTokenBase):
    """
    Custom Pytorch Lightning module to train text classification models
    using a transformer layer followed by a simple classifier.
    WARNING: it is currently available only for binary classification.

    Parameters
    ----------
    bert_model: str
        Path of the CamembertModel to load.
    num_classes: int
        Number of label classes.
    finetune: bool
        Whether to finetune (i.e modify the last layers) the BERT model or not.
    dropout: float
        Dropout probability for the classifier.
    lr: float
        Learning rate to reach after the warmup steps and before the decay.
    class_weights: Optional[Dict[str, float]]
        Weights to apply to each class.
    num_epoch: int
        Number of training epochs (only useful for warmup)
    num_warmup_epoch: int
        Number of warmup epoch for the learning rate.
    scheduler: dict
        Parameters of scheduler. Default= dict(scheduler='ReduceLROnPlateau',
        factor=.5,patience=2,verbose=True)
    """

    def __init__(
        self,
        camembert_model_path: str,
        num_classes: int = 2,
        finetune: bool = True,
        dropout: float = 0.1,
        config_optimizer={"lr": 1e-4},
        class_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        num_epoch: int = 10,
        num_warmup_epoch: int = 1,
        scheduler: dict = dict(
            scheduler="ReduceLROnPlateau", factor=0.5, patience=2, verbose=True
        ),
        data_dir="../../data/datasets/annotation_results_clean_k1=35_k2=10",  # noqa: E501
    ):
        super().__init__(
            camembert_model_path=camembert_model_path,
            num_classes=num_classes,
            finetune=finetune,
            dropout=dropout,
            config_optimizer=config_optimizer,
            class_weights=class_weights,
            batch_size=batch_size,
            num_epoch=num_epoch,
            num_warmup_epoch=num_warmup_epoch,
            scheduler=scheduler,
            data_dir=data_dir,
        )
        self.save_hyperparameters()

        self.classification_head = RobertaClassificationHead(
            dropout=dropout,
            hidden_size=self.bert.pooler.dense.out_features,
            num_classes=num_classes,
        )
