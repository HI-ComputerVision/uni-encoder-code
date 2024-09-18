import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNetLike(nn.Module):
    def __init__(self, ResidualBlock=ResidualBlock, num_input_features=1, num_frames_to_predict_for=2):
        super(ResNetLike, self).__init__()
        self.layer1 = self.make_layer(ResidualBlock, 192, 64, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 384 + 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 768 + 128, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 1536 + 256, 512, 2, stride=2)        
        self.squeeze = nn.Conv2d(512, 256, 1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleDict({
            "pose_0": nn.Conv2d(num_input_features * 256, 256, 3, 1, 1),
            "pose_1": nn.Conv2d(256, 256, 3, 1, 1),
            "pose_2": nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        })
        self.num_frames_to_predict_for = num_frames_to_predict_for
        
    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1))
        for stride in strides:
            layers.append(block(out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, features):
        res2, res3, res4, res5 = features['res2'], features['res3'], features['res4'], features['res5']
        out = self.layer1(res2)
        out = self.layer2(torch.cat([out, res3], dim=1))
        out = self.layer3(torch.cat([out, res4], dim=1))
        out = self.layer4(torch.cat([out, res5], dim=1))
        out = self.relu(self.squeeze(out))
        for i in range(3):
            out = self.convs[f"pose_{i}"](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation