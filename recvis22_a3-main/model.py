import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.resnet1 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        self.resnet2 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.resnet3 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        self.resnet4 = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=1),
            nn.BatchNorm2d(512)
        )

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, nclasses)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.bn1(x)
        x = x + self.resnet1(x)
        x = self.dropout1 (F.max_pool2d(F.relu(self.conv2(x)), 2) )
        x = self.bn2(x)
        x = x + self.resnet2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        x = self.bn3(x)
        x= x + self.resnet3(x)
        x = F.max_pool2d (F.relu(self.conv4(x)),2)
        x = self.bn4(x)
        x = x + self.resnet4(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.dropout2( F.relu(self.fc1(x)) ) 
        return  self.fc2(x) 
    


class AttentionModule(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
       
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
   
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
       
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
       
        return x

class BirdClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(BirdClassifier, self).__init__()
       
        # Use EfficientNet as backbone
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
       
        # Get feature dimensions
        in_features = self.backbone.classifier[1].in_features
       
        # Replace classifier with custom head
        self.backbone.classifier = nn.Identity()
       
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes)
        )
       
        # Attention module
        self.attention = AttentionModule(1536)  # EfficientNet-B3 features
       
        # Initialize weights
        self._initialize_weights()
   
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
   
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
       
        # Apply attention (you might need to reshape for attention)
        if features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)
       
        features = self.attention(features)
        features = features.mean([2, 3])  # Global average pooling
       
        # Classification
        output = self.classifier(features)
        return output

class LayerResidual(nn.Module):
    def __init__(self):
        super(LayerResidual, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride =2, padding = 3)
        self.batchnorm0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm10 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm12 = nn.BatchNorm2d(64)
        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm13 = nn.BatchNorm2d(64)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm15 = nn.BatchNorm2d(64)

        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.conv20 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm22 = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm23 = nn.BatchNorm2d(128)
        self.conv24 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm24 = nn.BatchNorm2d(128)
        self.conv25 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm25 = nn.BatchNorm2d(128)
        self.conv26 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm26 = nn.BatchNorm2d(128)
        self.conv27 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm27 = nn.BatchNorm2d(128)

        self.shortcut3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.conv30 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.batchnorm30 = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm33 = nn.BatchNorm2d(256)
        self.conv34 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm36 = nn.BatchNorm2d(256)
        self.conv37 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm39 = nn.BatchNorm2d(256)
        self.conv310 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm310 = nn.BatchNorm2d(256)
        self.conv311 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm311 = nn.BatchNorm2d(256)

        self.shortcut4 = nn.Conv2d(256, 512, kernel_size=1, stride=2)


        self.conv40 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.batchnorm40 = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.batchnorm42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.batchnorm43 = nn.BatchNorm2d(512)
        self.conv44 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.batchnorm44 = nn.BatchNorm2d(512)
        self.conv45 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)
        self.batchnorm45 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 20)




    def forward(self, x):
        x = F.relu(self.batchnorm0(self.conv0(x)))
        x = self.pool0(x)

        y = F.relu(self.batchnorm10(self.conv10(x)))
        y = F.relu(self.batchnorm11(self.conv11(y)))
        x = y + x
        y = F.relu(self.batchnorm12(self.conv12(x)))
        y = F.relu(self.batchnorm13(self.conv13(y)))
        x = y + x

        y = F.relu(self.batchnorm14(self.conv14(x)))
        y = F.relu(self.batchnorm15(self.conv15(y)))
        x = x + y


        y = F.relu(self.batchnorm20(self.conv20(x)))
        y = F.relu(self.batchnorm21(self.conv21(y)))
        x = y + self.shortcut2(x)
        y = F.relu(self.batchnorm22(self.conv22(x)))
        y = F.relu(self.batchnorm23(self.conv23(y)))
        x = y + x
        y = F.relu(self.batchnorm24(self.conv24(x)))
        y = F.relu(self.batchnorm25(self.conv25(y)))
        x = y + x
        y = F.relu(self.batchnorm26(self.conv26(x)))
        y = F.relu(self.batchnorm27(self.conv27(y)))
        x = y + x    

        y = F.relu(self.batchnorm30(self.conv30(x)))
        y = F.relu(self.batchnorm31(self.conv31(y)))
        x = y + self.shortcut3(x)
        y = F.relu(self.batchnorm32(self.conv32(x)))
        y = F.relu(self.batchnorm33(self.conv33(y)))
        x = y + x
        y = F.relu(self.batchnorm34(self.conv34(x)))
        y = F.relu(self.batchnorm35(self.conv35(y)))
        x = y + x
        y = F.relu(self.batchnorm36(self.conv36(x)))
        y = F.relu(self.batchnorm37(self.conv37(y)))
        x = y + x
        y = F.relu(self.batchnorm38(self.conv38(x)))
        y = F.relu(self.batchnorm39(self.conv39(y)))
        x = y + x
        y = F.relu(self.batchnorm310(self.conv310(x)))
        y = F.relu(self.batchnorm311(self.conv311(y)))
        x = y + x
        
        y = F.relu(self.batchnorm40(self.conv40(x)))
        y = F.relu(self.batchnorm41(self.conv41(y)))
        x = y + self.shortcut4(x)
        y = F.relu(self.batchnorm42(self.conv42(x)))
        y = F.relu(self.batchnorm43(self.conv43(y)))
        x = y + x
        y = F.relu(self.batchnorm44(self.conv44(x)))
        y = F.relu(self.batchnorm45(self.conv45(y)))        
        x = y + x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 