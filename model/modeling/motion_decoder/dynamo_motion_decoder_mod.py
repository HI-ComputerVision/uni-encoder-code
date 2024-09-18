import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
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
        out = F.elu(out)
        
        return out

class MotionDecoderV2(nn.Module):

    def __init__(self, scales=range(4), num_input_images=2, out_dim=4):
        super(MotionDecoderV2, self).__init__()
        self.num_inp_feat = [6, 64, 192, 384, 768, 1536]
        self.out_dim = out_dim
        self.scales = scales
        self.layer0 = self._make_fusion_layer(ResidualBlock, 192, 64, 2, stride=1)
        self.layer1 = self._make_fusion_layer(ResidualBlock, 64, 64, 2, stride=2)
        self.layer2 = self._make_fusion_layer(ResidualBlock, 192 + 64, 64, 2, stride=2)
        self.layer3 = self._make_fusion_layer(ResidualBlock, 384 + 64, 128, 2, stride=2)
        self.layer4 = self._make_fusion_layer(ResidualBlock, 768 + 128, 256, 2, stride=2)
        
        self.conv0, self.squeeze0 = self._make_layer(0)
        self.conv1, self.squeeze1 = self._make_layer(1)
        self.conv2, self.squeeze2 = self._make_layer(2)
        self.conv3, self.squeeze3 = self._make_layer(3)
        self.conv4, self.squeeze4 = self._make_layer(4)
        self.conv5, self.squeeze5 = self._make_layer(5)
        self.res_trans_conv = nn.Conv2d(6, out_dim, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    
    def _make_fusion_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1))
        for stride in strides:
            layers.append(block(out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def _make_layer(self, stage):
        conv = nn.Sequential(
            nn.Conv2d(self.num_inp_feat[stage] + self.out_dim, self.num_inp_feat[stage], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.num_inp_feat[stage], self.num_inp_feat[stage], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        squeeze = nn.Conv2d(self.num_inp_feat[stage]*2, self.out_dim, kernel_size=1, stride=1)
        return conv, squeeze

    def forward(self, pose_feat, ego_motion):
        '''
        pose_feat: [(B, 6, H, W), (B, 64, H//2, W//2), (B, 64, H//4, W//4), (B, 128, H//8, W//8), (B, 256, H//16, W//16), (B, 512, H//32, W//32)]
        ego_motion: (B, 6, 1, 1)
        '''
        feat = pose_feat['motion_input']['full_res_input']
        feat0 = feat
        feat1 = F.interpolate(pose_feat['motion_input']['res2'].detach(), scale_factor=2, mode='bilinear', align_corners=False)
        feat1 = self.layer0(feat1)

        res_trans = self.res_trans_conv(100 * ego_motion)    # (B, out_dim, 1, 1)
        
        feat5 = pose_feat['motion_input']['res5']
        motion_field = F.interpolate(res_trans, size=feat5.shape[-2:], mode='bilinear', align_corners=False)
        x5a = self.conv5[:1](torch.cat([motion_field, feat5], dim=1))
        x5b = self.conv5[1:](x5a)
        out5 = self.squeeze5(torch.cat([x5a, x5b], dim=1)) + motion_field
        
        feat4 = pose_feat['motion_input']['res4']
        motion_field = F.interpolate(out5, size=feat4.shape[-2:], mode='bilinear', align_corners=False)
        x4a = self.conv4[:1](torch.cat([motion_field, feat4], dim=1))
        x4b = self.conv4[1:](x4a)
        out4 = self.squeeze4(torch.cat([x4a, x4b], dim=1)) + motion_field
        
        feat3 = pose_feat['motion_input']['res3']
        motion_field = F.interpolate(out4, size=feat3.shape[-2:], mode='bilinear', align_corners=False)
        x3a = self.conv3[:1](torch.cat([motion_field, feat3], dim=1))
        x3b = self.conv3[1:](x3a)
        out3 = self.squeeze3(torch.cat([x3a, x3b], dim=1)) + motion_field

        feat2 = pose_feat['motion_input']['res2']
        motion_field = F.interpolate(out3, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
        x2a = self.conv2[:1](torch.cat([motion_field, feat2], dim=1))
        x2b = self.conv2[1:](x2a)
        out2 = self.squeeze2(torch.cat([x2a, x2b], dim=1)) + motion_field

        motion_field = F.interpolate(out2, size=feat1.shape[-2:], mode='bilinear', align_corners=False)
        x1a = self.conv1[:1](torch.cat([motion_field, feat1], dim=1))
        x1b = self.conv1[1:](x1a)
        out1 = self.squeeze1(torch.cat([x1a, x1b], dim=1)) + motion_field

        motion_field = F.interpolate(out1, size=feat0.shape[-2:], mode='bilinear', align_corners=False)
        x0a = self.conv0[:1](torch.cat([motion_field, feat0], dim=1))
        x0b = self.conv0[1:](x0a)
        out0 = self.squeeze0(torch.cat([x0a, x0b], dim=1)) + motion_field

        outputs = dict()
        for scale in self.scales:
            if self.out_dim == 1:       # used to predict binary motion mask
                outputs[('motion_prob', scale)] =    0.005 * locals()[f'out{scale}']
                outputs[('motion_mask', scale)] =    torch.sigmoid(0.005 * locals()[f'out{scale}'])
            elif self.out_dim == 3:     #  used to predict complete 3D flow
                outputs[('complete_flow', scale)] =    0.005 * locals()[f'out{scale}']
            else:
                raise Exception(f'out_dim={self.out_dim} not excepted.')

        return outputs

