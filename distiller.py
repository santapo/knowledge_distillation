
import pytorch_lightning as pl

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import accuracy

from model import get_model

class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        student_name,
        teacher_name,
        lr=0.05,
        optimizer_name='adam'
    ):
        super().__init__()
        
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.student_model = get_model(model_name=student_name)
        self.teacher_model = get_model(model_name=teacher_name)
        self.teacher_model.eval()
    
    def configure_optimizers(self):

        trainable_parameters = self.model.parameters()

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(trainable_parameters, lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer_name} is not defined')

        return optimizer

    def loss(self, outputs, labels, teacher_outputs):
        alpha = 0.4
        loss = nn.KLDivLoss()(outputs, teacher_outputs) * alpha \
                + F.nll_loss(outputs, labels) * (1 - alpha)
        return loss

    def forward(self, x, mode):
        if mode == 'teacher':
            out = self.teacher_model(x)
        if mode == 'student':
            out = self.student_model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_student = self.forward(x, 'student')
        y_teacher = self.forward(x, 'teacher')
        
        loss = self.loss(y_student, y, y_teacher)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.forward(x, 'student')
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    