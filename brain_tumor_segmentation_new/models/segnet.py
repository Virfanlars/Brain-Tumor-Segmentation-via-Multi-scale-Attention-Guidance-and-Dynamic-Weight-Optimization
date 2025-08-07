import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SegNet, self).__init__()
        
        # Load pretrained VGG16 features
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        
        # Encoder blocks
        self.enc1 = nn.Sequential(*features[0:5])   # 64
        self.enc2 = nn.Sequential(*features[5:10])  # 128
        self.enc3 = nn.Sequential(*features[10:17]) # 256
        self.enc4 = nn.Sequential(*features[17:24]) # 512
        self.enc5 = nn.Sequential(*features[24:])   # 512
        
        # Decoder blocks with corresponding feature channels
        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x1_size = x1.size()
        x1_pool, ind1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        
        x2 = self.enc2(x1_pool)
        x2_size = x2.size()
        x2_pool, ind2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        
        x3 = self.enc3(x2_pool)
        x3_size = x3.size()
        x3_pool, ind3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        
        x4 = self.enc4(x3_pool)
        x4_size = x4.size()
        x4_pool, ind4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        
        x5 = self.enc5(x4_pool)
        x5_size = x5.size()
        x5_pool, ind5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder
        y5 = F.max_unpool2d(x5_pool, ind5, kernel_size=2, stride=2, output_size=x5_size)
        y5 = self.dec5(y5)
        
        y4 = F.max_unpool2d(y5, ind4, kernel_size=2, stride=2, output_size=x4_size)
        y4 = self.dec4(y4)
        
        y3 = F.max_unpool2d(y4, ind3, kernel_size=2, stride=2, output_size=x3_size)
        y3 = self.dec3(y3)
        
        y2 = F.max_unpool2d(y3, ind2, kernel_size=2, stride=2, output_size=x2_size)
        y2 = self.dec2(y2)
        
        y1 = F.max_unpool2d(y2, ind1, kernel_size=2, stride=2, output_size=x1_size)
        y1 = self.dec1(y1)
        
        return y1 