import pytorch_lightning as pl

class ClassificationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        