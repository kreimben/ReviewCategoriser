import os

import lightning as L
import torch
from torch.utils.data import DataLoader


class RoBERTaFineTuner(L.LightningModule):
    def __init__(self, model,
                 train_dataset, valid_dataset, test_dataset,
                 learning_rate=1e-5, batch_size=32, device='cuda'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        # self.learning_rate = 1e-5
        # self.batch_size = 32

        # dataset
        self.train_dataset = train_dataset
        self.val_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.param = {
            'num_workers': os.cpu_count() - 1,
            'persistent_workers': True,
            'generator': torch.Generator(device=device),
            'pin_memory': True
        }

    def __accuracy(self, preds, labels):
        _, predicted_classes = torch.max(preds, dim=1)
        correct = (predicted_classes == labels).sum().item()
        accuracy = correct / len(labels) * 100.0
        return accuracy

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def __step(self, log_name: str, batch, batch_idx, *args, **kwargs):
        """
        Just for validation and test step!!!
        :param log_name: `val` or `test`
        :param batch:
        :param batch_idx:
        :param args: Added just in case.
        :param kwargs: Added just in case.
        :return:
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = self(input_ids, attention_mask=attention_mask, labels=labels)

            pred = outputs.logits
            acc = self.__accuracy(pred, labels)

        # Log loss and accuracy
        self.log(f'{log_name}_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{log_name}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask=attention_mask, labels=labels)

        pred = outputs.logits
        acc = self.__accuracy(pred, labels)

        # Log loss and accuracy
        self.log(f'train_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        return self.__step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.__step('test', batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size,
                          **self.param)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, **self.param)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, **self.param)  # | self.batch_size
