import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))

        # Move only the tensors to the correct device (ignore the 'sentence' field)
        input_ids = val_batch["input_ids"].to(pl_module.device)
        attention_mask = val_batch["attention_mask"].to(pl_module.device)
        labels = val_batch["label"].to(pl_module.device)

        sentences = val_batch["sentence"]

        outputs = pl_module(input_ids, attention_mask)

        preds = torch.argmax(outputs.logits, 1)
        # labels = val_batch["label"]

        # Convert to CPU for logging
        df = pd.DataFrame(
            {
                "Sentence": sentences,
                "Label": labels.cpu().numpy(),
                "Predicted": preds.cpu().numpy(),
            }
        )


        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        # limit_train_batches=0.25,
        # limit_val_batches=0.25
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
