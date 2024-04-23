import torch
import torchvision
import torch.nn as nn

class VGG16BNFeats(torch.nn.Module):
    # "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    # conv1_1: conv, bn, relu 3
    # conv1_2: conv, bn, relu 6
    # "M": maxpool 7
    # conv2_1: conv, bn, relu 10
    # conv2_2: conv, bn, relu 13
    # "M": maxpool 14
    # conv3_1: conv, bn, relu 17
    # conv3_2: conv, bn, relu 20   
    # conv3_3: conv, bn, relu 23
    # "M": maxpool 24
    # conv4_1: conv, bn, relu 27
    # conv4_2: conv, bn, relu 30   
    # conv4_3: conv, bn, relu 33
    # "M": maxpool 34
    # conv5_1: conv, bn, relu 37
    # conv5_2: conv, bn, relu 40   
    # conv5_3: conv, bn, relu 43
    # "M": maxpool 44

    def __init__(self, requires_grad=True):
        super(VGG16BNFeats, self).__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True)

        vgg_features = vgg.features
        self.conv1_2 = vgg_features[:4]
        self.conv2_2 = vgg_features[4:11]
        self.conv3_3 = vgg_features[11:21]
        self.conv4_3 = vgg_features[21:31]
        self.conv5_3 = vgg_features[31:41]
        self.conv_last = vgg_features[41:44]
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
        ) 
    def forward(self, img):
        # conv_feats = []
        conv1_2 = self.conv1_2(img)
        conv2_2 = self.conv2_2(conv1_2)
        conv3_3 = self.conv3_3(conv2_2)
        conv4_3 = self.conv4_3(conv3_3)
        conv5_3 = self.conv5_3(conv4_3)
        conv_last = self.conv_last(conv5_3)
        feat_global = conv_last
        feat_global = self.avgpool(conv_last)
        feat_global = torch.flatten(feat_global, 1)
        feat_global = self.classifier(feat_global)
        return [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3], feat_global