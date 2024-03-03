import torch

from torch.nn import functional as F

from torch.nn import (
    Conv2d,
    Linear,
    Flatten,
    MaxPool2d,
    AdaptiveAvgPool2d,
    ZeroPad2d,
)

from ssd_layers import Normalize, PriorBox

class SSD300v2(torch.nn.Module):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image (3, 300, 300).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    def __init__(self, input_shape, num_classes=21):
        super(SSD300v2, self).__init__()

        self.num_classes = num_classes

        # Block 1
        self.conv1_1 = Conv2d(3, 64, (3, 3), padding=(1, 1))
        self.conv1_2 = Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2

        self.conv2_1 = Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.conv2_2 = Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 3
        self.conv3_1 = Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.conv3_2 = Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.conv3_3 = Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.conv3_3z = ZeroPad2d(padding=(0, 1, 0, 1))
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 4
        self.conv4_1 = Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.conv4_2 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv4_3 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 5
        self.conv5_1 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv5_2 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.conv5_3 = Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.pool5 = MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # FC6
        self.fc6 = Conv2d(512, 1024, (3, 3), dilation=(6, 6), padding=(6, 6))

        # FC7
        self.fc7 = Conv2d(1024, 1024, (1, 1), padding=(0, 0))

        # Block 6
        self.conv6_1 = Conv2d(1024, 256, (1, 1), padding=(0, 0))
        self.conv6_1z = ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv6_2 = Conv2d(256, 512, (3, 3), stride=(2, 2))

        # Block 7
        self.conv7_1 = Conv2d(512, 128, (1, 1), padding=(0, 0))
        self.conv7_1z = ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv7_2 = Conv2d(128, 256, (3, 3), stride=(2, 2))

        # Block 8
        self.conv8_1 = Conv2d(256, 128, (1, 1), padding=(0, 0))
        self.conv8_2 = Conv2d(128, 256, (3, 3), padding=(1, 1), stride=(2, 2))

        #self.pool6 = AdaptiveAvgPool2d((1, 1))

        img_size = (input_shape[1], input_shape[0])

        # predictions from conv4_3
        num_priors = 3

        self.conv4_3_norm = Normalize(20, name='conv4_3_norm', input_shape=(-1, 512, -1, -1))
        self.conv4_3_norm_mbox_loc = Conv2d(512, num_priors * 4, (3, 3), padding=(1, 1))
        self.conv4_3_norm_mbox_loc_flat = Flatten()
        self.conv4_3_norm_mbox_conf = Conv2d(512, num_priors * num_classes, (3, 3), padding=(1, 1))
        self.conv4_3_norm_mbox_conf_flat = Flatten()

        self.conv4_3_norm_mbox_priorbox = PriorBox(img_size, 30.0,
                                                   aspect_ratios=[2],
                                                   variances=[0.1, 0.1, 0.2, 0.2])


        # predictions from fc7

        num_priors = 6
        self.fc7_mbox_conf = Conv2d(1024, num_priors * num_classes, (3, 3), padding=(1, 1))
        self.fc7_mbox_conf_flat = Flatten()
        self.fc7_mbox_loc = Conv2d(1024, num_priors * 4, (3, 3), padding=(1, 1))
        self.fc7_mbox_loc_flat = Flatten()

        self.fc7_mbox_priorbox = PriorBox(img_size, 60.0,
                                     max_size=114.0,
                                     aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2])

        # predictions from conv6_2

        num_priors = 6
        self.conv6_2_mbox_conf = Conv2d(512, num_priors * num_classes, (3, 3), padding=(1, 1))
        self.conv6_2_mbox_conf_flat = Flatten()
        self.conv6_2_mbox_loc = Conv2d(512, num_priors * 4, (3, 3), padding=(1, 1))
        self.conv6_2_mbox_loc_flat = Flatten()

        self.conv6_2_mbox_priorbox = PriorBox(img_size, 114.0,
                                         max_size=168.0,
                                         aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2])

        # predictions from conv7_2

        self.conv7_2_mbox_conf = Conv2d(256, num_priors * num_classes, (3, 3), padding=(1, 1))
        self.conv7_2_mbox_conf_flat = Flatten()
        self.conv7_2_mbox_loc = Conv2d(256, num_priors * 4, (3, 3), padding=(1, 1))
        self.conv7_2_mbox_loc_flat = Flatten()

        self.conv7_2_mbox_priorbox = PriorBox(img_size, 168.0,
                                         max_size=222.0,
                                         aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2])

        # predictions from conv8_2

        self.conv8_2_mbox_conf = Conv2d(256, num_priors * num_classes, (3, 3), padding=(1, 1))
        self.conv8_2_mbox_conf_flat = Flatten()
        self.conv8_2_mbox_loc = Conv2d(256, num_priors * 4, (3, 3), padding=(1, 1))
        self.conv8_2_mbox_loc_flat = Flatten()

        self.conv8_2_mbox_priorbox = PriorBox(img_size, 222.0,
                                         max_size=276.0,
                                         aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2])

        # predictions from pool6

        self.pool6_mbox_loc_flat = Linear(256, num_priors * 4)
        self.pool6_mbox_conf_flat = Linear(256, num_priors * num_classes)

        self.pool6_mbox_priorbox = PriorBox(img_size, 276.0,
                                       max_size=330.0,
                                       aspect_ratios=[2, 3],
                                       variances=[0.1, 0.1, 0.2, 0.2])


    def forward(self, input_object):

        input_shape = input_object.shape

        def conv_activation(obj, conv, activ):
            return activ(conv(obj), inplace=True)

        conv1_1 = conv_activation(input_object, self.conv1_1, F.relu)
        conv1_2 = conv_activation(conv1_1, self.conv1_2, F.relu)
        pool1 = self.pool1(conv1_2)

        conv2_1 = conv_activation(pool1, self.conv2_1, F.relu)
        conv2_2 = conv_activation(conv2_1, self.conv2_2, F.relu)
        pool2 = self.pool2(conv2_2)

        conv3_1 = conv_activation(pool2, self.conv3_1, F.relu)
        conv3_2 = conv_activation(conv3_1, self.conv3_2, F.relu)
        conv3_3 = conv_activation(conv3_2, self.conv3_3, F.relu)
        conv3_3z = self.conv3_3z(conv3_3)
        pool3 = self.pool3(conv3_3z)

        conv4_1 = conv_activation(pool3, self.conv4_1, F.relu)
        conv4_2 = conv_activation(conv4_1, self.conv4_2, F.relu)
        conv4_3 = conv_activation(conv4_2, self.conv4_3, F.relu)
        pool4 = self.pool4(conv4_3)

        conv5_1 = conv_activation(pool4, self.conv5_1, F.relu)
        conv5_2 = conv_activation(conv5_1, self.conv5_2, F.relu)
        conv5_3 = conv_activation(conv5_2, self.conv5_3, F.relu)
        pool5 = self.pool5(conv5_3)

        fc6 = conv_activation(pool5, self.fc6, F.relu)
        fc7 = conv_activation(fc6, self.fc7, F.relu)

        conv6_1 = conv_activation(fc7, self.conv6_1, F.relu)
        conv6_1z = self.conv6_1z(conv6_1)
        conv6_2 = conv_activation(conv6_1z, self.conv6_2, F.relu)

        conv7_1 = conv_activation(conv6_2, self.conv7_1, F.relu)
        conv7_1z = self.conv7_1z(conv7_1) # zero padding
        conv7_2 = conv_activation(conv7_1z, self.conv7_2, F.relu)

        conv8_1 = conv_activation(conv7_2, self.conv8_1, F.relu)
        conv8_2 = conv_activation(conv8_1, self.conv8_2, F.relu)

        pool6 = conv8_2.mean(dim=(-2, -1))

        # predictions from conv4_3

        conv4_3_norm = self.conv4_3_norm(conv4_3)
        conv4_3_norm_mbox_loc = self.conv4_3_norm_mbox_loc(conv4_3_norm)
        conv4_3_norm_mbox_loc = conv4_3_norm_mbox_loc.permute(0, 2, 3, 1)
        conv4_3_norm_mbox_loc_flat = self.conv4_3_norm_mbox_loc_flat(conv4_3_norm_mbox_loc)
        conv4_3_norm_mbox_conf = self.conv4_3_norm_mbox_conf(conv4_3_norm)
        conv4_3_norm_mbox_conf = conv4_3_norm_mbox_conf.permute(0, 2, 3, 1)
        conv4_3_norm_mbox_conf_flat = self.conv4_3_norm_mbox_conf_flat(conv4_3_norm_mbox_conf)
        conv4_3_norm_mbox_priorbox = self.conv4_3_norm_mbox_priorbox(conv4_3_norm)

        # predictions from fc7

        fc7_mbox_loc = self.fc7_mbox_loc(fc7)
        fc7_mbox_loc = fc7_mbox_loc.permute(0, 2, 3, 1)
        fc7_mbox_loc_flat = self.fc7_mbox_loc_flat(fc7_mbox_loc)
        fc7_mbox_conf = self.fc7_mbox_conf(fc7)
        fc7_mbox_conf = fc7_mbox_conf.permute(0, 2, 3, 1)
        fc7_mbox_conf_flat = self.fc7_mbox_conf_flat(fc7_mbox_conf)
        fc7_mbox_priorbox = self.fc7_mbox_priorbox(fc7)

        # predictions from conv6_2

        conv6_2_mbox_loc = self.conv6_2_mbox_loc(conv6_2)
        conv6_2_mbox_loc = conv6_2_mbox_loc.permute(0, 2, 3, 1)
        conv6_2_mbox_loc_flat = self.conv6_2_mbox_loc_flat(conv6_2_mbox_loc)
        conv6_2_mbox_conf = self.conv6_2_mbox_conf(conv6_2)
        conv6_2_mbox_conf = conv6_2_mbox_conf.permute(0, 2, 3, 1)
        conv6_2_mbox_conf_flat = self.conv6_2_mbox_conf_flat(conv6_2_mbox_conf)
        conv6_2_mbox_priorbox = self.conv6_2_mbox_priorbox(conv6_2)

        # predictions from conv7_2

        conv7_2_mbox_loc = self.conv7_2_mbox_loc(conv7_2)
        conv7_2_mbox_loc = conv7_2_mbox_loc.permute(0, 2, 3, 1)
        conv7_2_mbox_loc_flat = self.conv7_2_mbox_loc_flat(conv7_2_mbox_loc)
        conv7_2_mbox_conf = self.conv7_2_mbox_conf(conv7_2)
        conv7_2_mbox_conf = conv7_2_mbox_conf.permute(0, 2, 3, 1)
        conv7_2_mbox_conf_flat = self.conv7_2_mbox_conf_flat(conv7_2_mbox_conf)
        conv7_2_mbox_priorbox = self.conv7_2_mbox_priorbox(conv7_2)

        # predictions from conv8_2

        conv8_2_mbox_loc = self.conv8_2_mbox_loc(conv8_2)
        conv8_2_mbox_loc = conv8_2_mbox_loc.permute(0, 2, 3, 1)
        conv8_2_mbox_loc_flat = self.conv8_2_mbox_loc_flat(conv8_2_mbox_loc)
        conv8_2_mbox_conf = self.conv8_2_mbox_conf(conv8_2)
        conv8_2_mbox_conf = conv8_2_mbox_conf.permute(0, 2, 3, 1)
        conv8_2_mbox_conf_flat = self.conv8_2_mbox_conf_flat(conv8_2_mbox_conf)
        conv8_2_mbox_priorbox = self.conv8_2_mbox_priorbox(conv8_2)

        # predictions from pool6

        pool6_mbox_loc_flat = self.pool6_mbox_loc_flat(pool6.view(input_object.size(0), -1))
        pool6_mbox_conf_flat = self.pool6_mbox_conf_flat(pool6.view(input_object.size(0), -1))
        pool6_mbox_priorbox = self.pool6_mbox_priorbox(pool6.view(input_object.size(0), -1, 1, 1))

        # gather all predictions

        mbox_loc = torch.cat(
            [
                conv4_3_norm_mbox_loc_flat,
                fc7_mbox_loc_flat,
                conv6_2_mbox_loc_flat,
                conv7_2_mbox_loc_flat,
                conv8_2_mbox_loc_flat,
                pool6_mbox_loc_flat
            ],
            dim=1
        )

        mbox_conf = torch.cat(
            [
                conv4_3_norm_mbox_conf_flat,
                fc7_mbox_conf_flat,
                conv6_2_mbox_conf_flat,
                conv7_2_mbox_conf_flat,
                conv8_2_mbox_conf_flat,
                pool6_mbox_conf_flat
            ],
            dim=1
        )

        mbox_priorbox = torch.cat(
            [
                conv4_3_norm_mbox_priorbox,
                fc7_mbox_priorbox,
                conv6_2_mbox_priorbox,
                conv7_2_mbox_priorbox,
                conv8_2_mbox_priorbox,
                pool6_mbox_priorbox
            ],
            dim=1
        )

        num_boxes = mbox_loc.shape[-1] // 4
        mbox_loc = mbox_loc.reshape((input_shape[0], num_boxes, 4))
        mbox_conf = mbox_conf.reshape((input_shape[0], num_boxes, self.num_classes))
        mbox_conf = F.softmax(mbox_conf, dim=-1)

        predictions = torch.cat(
            [
                mbox_loc,
                mbox_conf,
                mbox_priorbox
            ],
            dim=2
        )
        return predictions

if __name__ == '__main__':
    from torchsummary import summary
    model = SSD300v2((300, 300, 3))
    summary(model, (3, 300, 300))
    obj = torch.randn(1, 3, 300, 300)
    print(model(obj).shape)
