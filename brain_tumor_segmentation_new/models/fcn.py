import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(FCN8s, self).__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        
        self.features3 = nn.Sequential(*features[0:17])    # pool3
        self.features4 = nn.Sequential(*features[17:24])   # pool4
        self.features5 = nn.Sequential(*features[24:])     # pool5
        
        # FCN head
        self.fcn5 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1)
        )
        
        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Original input size
        input_size = x.size()[2:]
        
        # VGG features
        pool3 = self.features3(x)      # 1/8
        pool4 = self.features4(pool3)  # 1/16
        pool5 = self.features5(pool4)  # 1/32
        
        # FCN head
        score5 = self.fcn5(pool5)
        
        # Upsample score5 to match pool4 size
        score5_up = F.interpolate(score5, size=pool4.size()[2:], mode='bilinear', align_corners=True)
        
        # Add skip connection from pool4
        score4 = self.score_pool4(pool4)
        score4 = score4 + score5_up
        
        # Upsample score4 to match pool3 size
        score4_up = F.interpolate(score4, size=pool3.size()[2:], mode='bilinear', align_corners=True)
        
        # Add skip connection from pool3
        score3 = self.score_pool3(pool3)
        score3 = score3 + score4_up
        
        # Upsample to original size
        out = F.interpolate(score3, size=input_size, mode='bilinear', align_corners=True)
        
        return out 