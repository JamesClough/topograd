""" Pytorch models used in topograd project """

import torch
from torch import nn
from torch.nn import functional as F

class Segmenter_Unet(nn.Module):
    def __init__(self, img_dim, num_filters=16):
        super(Segmenter_Unet, self).__init__()
        self.img_dim = img_dim
        self.num_filters = num_filters

        # have series of convs down, using maxpools
        # and then convs back up again
        # just upsample on the way back up
        self.conv1_1 = nn.Conv2d(1, self.num_filters, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(self.num_filters, self.num_filters,   3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(self.num_filters, self.num_filters*2, 3, stride=2, padding=1)
        # downsampled
        self.conv2_1 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(self.num_filters*2, self.num_filters*4, 3, stride=2, padding=1)
        # downsampled
        self.conv3_1 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)

        # upsampled - will have a num_filters * 2 concatted in
        self.conv4_1 = nn.Conv2d(self.num_filters*6, self.num_filters*2, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)

        # upsampled - will have num_filters * 1 concatted in
        self.conv5_1 = nn.Conv2d(self.num_filters*3, self.num_filters*1, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(self.num_filters*1, self.num_filters*1, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(self.num_filters*1, self.num_filters*1, 3, stride=1, padding=1)

        # finish with 1x1 conv
        self.conv_final = nn.Conv2d(self.num_filters*1, 1, 1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x_1 = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x_1))

        x = F.relu(self.conv2_1(x))
        x_2 = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x_2))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))

        x = torch.cat([F.interpolate(x, scale_factor=2), x_2], dim=1)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = torch.cat([F.interpolate(x, scale_factor=2), x_1], dim=1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = torch.sigmoid(self.conv_final(x))
        return x

class MNIST_classifier(nn.Module):
    def __init__(self, img_dim, num_filters=16, num_classes=10):
        super(MNIST_classifier, self).__init__()
        self.img_dim = img_dim
        self.num_filters = num_filters
        self.num_classes = num_classes

        # have series of convs down, using maxpools
        # and then convs back up again
        # just upsample on the way back up
        self.conv1_1 = nn.Conv2d(1, self.num_filters, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(self.num_filters, self.num_filters,   3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(self.num_filters, self.num_filters*2, 3, stride=2, padding=1)
        # downsampled
        self.conv2_1 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(self.num_filters*2, self.num_filters*4, 3, stride=2, padding=1)
        # downsampled
        self.conv3_1 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)

        self.low_res_img_dim = self.img_dim // 4
        self.final_conv_num_filters = self.num_filters*4
        self.fc_1 = nn.Linear(self.low_res_img_dim**2 * self.final_conv_num_filters, self.final_conv_num_filters)
        self.fc_2 = nn.Linear(self.final_conv_num_filters, self.final_conv_num_filters)
        self.fc_3 = nn.Linear(self.final_conv_num_filters, self.final_conv_num_filters)

        self.fc_final = nn.Linear(self.final_conv_num_filters, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))

        x = x.view(-1, self.low_res_img_dim**2 * self.final_conv_num_filters)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_final(x)
        return x
