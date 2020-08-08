import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from dcn import DCNv1

class Conv3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, is_pool=False, is_dfe=False):
        super(Conv3x3Block, self).__init__()
        if is_dfe:
            self.block = [DCNv1(in_channels, out_channels, kernel_size=3, padding=padding, stride=1)]
        else:
            self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=1)]
        self.block += [nn.BatchNorm2d(out_channels, momentum=0.1, affine=True),
                       nn.ReLU(inplace=True)]
 
        if is_pool:
            self.block.append(nn.MaxPool2d(2))           
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        out = self.block(x)
        return out

class Conv4(nn.Module):
    def __init__(self, feature_dim):
        super(Conv4, self).__init__()
        self.layer1 = Conv3x3Block(3, feature_dim)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim)
        self.layer3 = Conv3x3Block(feature_dim, feature_dim, padding=1, is_pool=True, is_dfe=True)
        self.layer4 = Conv3x3Block(feature_dim, feature_dim, padding=1, is_pool=True, is_dfe=True)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out 

class DCANetwork(nn.Module):
    def __init__(self, feature_dim):
        super(DCANetwork, self).__init__()
        self.in_channels = feature_dim
       
        self.correlation_layer = DCAModule(self.in_channels)
        self.layer1 = Conv3x3Block(self.in_channels*6, feature_dim, is_pool=True)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim, is_pool=True)
        self.fc = nn.Sequential(nn.Linear(feature_dim*3*3, 8),
                                nn.ReLU(),
                                nn.Linear(8, 1),
                                nn.Sigmoid())


    def forward(self, x1, x2): 
        out1, out2 = self.correlation_layer(x1, x2)

        out = self.layer1(out1)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out1 = self.fc(out)

        out = self.layer1(out2)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out2 = self.fc(out)

        return [out1, out2]

class DCAModule(nn.Module):
    def __init__(self, feature_dim, size=20):
        super(DCAModule, self).__init__()

        self.feature_dim = feature_dim

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
            kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
                   kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        spatial_shape = x1.shape[-2:]
        
        theta_x1 = self.theta(x1).view(batch_size, self.feature_dim, -1)
        theta_x2 = self.theta(x2).view(batch_size, self.feature_dim, -1)
        phi_x1 = theta_x1
        phi_x2 = theta_x2
        theta_x1 = theta_x1.permute(0, 2, 1)
        theta_x2 = theta_x2.permute(0, 2, 1)
 
        theta_x1 = F.normalize(theta_x1, dim=2)
        theta_x2 = F.normalize(theta_x2, dim=2)
        phi_x1 = F.normalize(phi_x1, dim=1)
        phi_x2 = F.normalize(phi_x2, dim=1)

        g_x1 = x1.view(batch_size, self.feature_dim, -1).permute(0, 2, 1)
        g_x2 = x2.view(batch_size, self.feature_dim, -1).permute(0, 2, 1)

        out1 = [x1, x2]
        out2 = [x2, x1]

        cross1 = torch.matmul(theta_x2, phi_x1)
        cross2 = cross1.permute(0, 2, 1)
        y_cross1 = torch.matmul(cross1, g_x1).permute(0, 2, 1).contiguous()
        y_cross1 = y_cross1.view(batch_size, self.feature_dim, *spatial_shape)
        y_cross2 = torch.matmul(cross2, g_x2).permute(0, 2, 1).contiguous()
        y_cross2 = y_cross2.view(batch_size, self.feature_dim, *spatial_shape)

        W_y_cross1 = self.W(y_cross1)
        W_y_cross2 = self.W(y_cross2)

        out1 += [W_y_cross1, W_y_cross2]
        out2 += [W_y_cross2, W_y_cross1]

        self1 = torch.matmul(theta_x1, phi_x1)
        self2 = torch.matmul(theta_x2, phi_x2)

        y_self1 = torch.matmul(self1, g_x1).permute(0, 2, 1).contiguous()
        y_self1 = y_self1.view(batch_size, self.feature_dim, *spatial_shape)

        y_self2 = torch.matmul(self2, g_x2).permute(0, 2, 1).contiguous()
        y_self2 = y_self2.view(batch_size, self.feature_dim, *spatial_shape)

        W_y_self1 = self.W(y_self1)
        W_y_self2 = self.W(y_self2)

        out1 += [W_y_self1, W_y_self2]
        out2 += [W_y_self2, W_y_self1]

        out1 = torch.cat(out1, 1)
        out2 = torch.cat(out2, 1)

        return out1, out2

