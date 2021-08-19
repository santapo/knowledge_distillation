import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()

        self.model = nn.Sequential([
            nn.Conv2d(3, 64, kernel_size=(5, 5)),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=3),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=3),
            nn.Flatten(),
            nn.Linear(2304, 10)
        ])

    def forward(self, x):
        return self.model(x)