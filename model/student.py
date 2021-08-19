import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.model = nn.Sequential([
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=3),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=3),
            nn.Flatten(),
            nn.Linear(1152, 10)
        ])

    def forward(self, x):
        return self.model(x)