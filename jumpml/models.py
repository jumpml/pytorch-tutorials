# © 2020 JumpML
import torch.nn as nn
import torch.nn.functional as F

def get_conv_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    x = get_conv_dim(input_size[0], kernel_size[0], stride, padding, dilation)
    if len(kernel_size) == 1:
        y = get_conv_dim(input_size[1], kernel_size[0], stride, padding, dilation)
    else:
        y = get_conv_dim(input_size[1], kernel_size[1], stride, padding, dilation)
    return (x,y)

def get_conv_dim(in_dim, kernel_size, stride=1, padding=0, dilation=1):
    return ( (in_dim + 2*padding - dilation * (kernel_size - 1) - 1) // stride  + 1)


class SpeechCommandsModel(nn.Module):
    def __init__(self, input_size=(1,64,101), C1=(32,8,20), C2=(8,4,10), MP=2, FC1=128, FC2=11):
        super(SpeechCommandsModel, self).__init__()
        
        # Model (size related) parameters
        self.input_numChan = input_size[0]
        self.input_size = (input_size[1], input_size[2])
        self.C1_numFilt = C1[0]
        self.C1_kernel_size  = (C1[1], C1[2])
        self.C2_numFilt = C2[0]
        self.C2_kernel_size  = (C2[1], C2[2])
        self.mp2d_size      = (MP,)
        self.fc1_in_dim = self.compute_fc1_input_dim()
        self.fc1_out_dim = FC1
        self.fc2_out_dim = FC2

        # Model definition
        self.conv1 = nn.Conv2d(input_size[0], C1[0], self.C1_kernel_size)
        self.bn1   = nn.BatchNorm2d(C1[0])
        self.conv2 = nn.Conv2d(C1[0], C2[0], self.C2_kernel_size)
        self.bn2   = nn.BatchNorm2d(C2[0])
        
        self.fc1 = nn.Linear(self.fc1_in_dim, self.fc1_out_dim)
        self.bn3   = nn.BatchNorm1d(self.fc1_out_dim)
        self.fc2 = nn.Linear(self.fc1_out_dim, self.fc2_out_dim)           # number of classes = 11

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), self.mp2d_size))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), self.mp2d_size))
        x = x.view(-1, self.fc1_in_dim)    # reshape
        x = F.relu(self.bn3(self.fc1(x)))
        #x = F.dropout(x, training=self.training)  # Apply dropout only during training
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def compute_fc1_input_dim(self):
        C1_size = get_conv_size(self.input_size, self.C1_kernel_size)
        MP1_size = get_conv_size(C1_size, self.mp2d_size, stride=self.mp2d_size[0])
        C2_size = get_conv_size(MP1_size, self.C2_kernel_size)
        MP2_size = get_conv_size(C2_size, self.mp2d_size, stride=self.mp2d_size[0])
        return(MP2_size[0] * MP2_size[1] * self.C2_numFilt)

