import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16,16,1)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.dp1 = nn.Dropout2d(0.01)

        self.conv6 = nn.Conv2d(16, 16, 3)
        self.bn6 = nn.BatchNorm2d(16)
        self.dp2 = nn.Dropout2d(0.01)

        self.conv7 = nn.Conv2d(16, 16 , 3)
        self.bn7 = nn.BatchNorm2d(16)
        self.dp3 = nn.Dropout2d(0.01)

        self.conv8 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(16)
        self.dp4 = nn.Dropout2d(0.01)

        self.conv9 = nn.Conv2d(16, 32 ,3, padding = 1)
        self.bn9 = nn.BatchNorm2d(32)
        self.dp5 = nn.Dropout2d(0.01)

        
        self.gap = nn.AvgPool2d(kernel_size=5)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = (self.bn1(F.relu(self.conv1(x))))
        x = (self.bn2(F.relu(self.conv2(x))))

        x = (self.pool1(self.bn3(F.relu(self.conv3(x)))))

        x = (self.bn4(F.relu(self.conv4(x))))

        x = self.dp1((self.bn5(F.relu(self.conv5(x)))))
        x = self.dp2((self.bn6(F.relu(self.conv6(x)))))
        x = self.dp3((self.bn7(F.relu(self.conv7(x)))))
        x = self.dp4((self.bn8(F.relu(self.conv8(x)))))
        x = self.dp5((self.bn9(F.relu(self.conv9(x)))))

        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.linear(x)
        return F.log_softmax(x)