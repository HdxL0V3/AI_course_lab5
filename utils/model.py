import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear, ReLU, BatchNorm2D, Embedding
import paddle.nn.functional as F
from data_load import *

class Basicblock(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Basicblock, self).__init__()
        self.stride = stride
        self.conv0 = Conv2D(in_channel, out_channel, 3, stride=stride, padding=1)
        self.conv1 = Conv2D(out_channel, out_channel, 3, stride=1, padding=1)
        self.conv2 = Conv2D(in_channel, out_channel, 1, stride=stride)
        self.bn0 = BatchNorm2D(out_channel)
        self.bn1 = BatchNorm2D(out_channel)
        self.bn2 = BatchNorm2D(out_channel)

    def forward(self, inputs):
        y = inputs
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.stride == 2:
            y = self.conv2(y)
            y = self.bn2(y)
        z = F.relu(x + y)
        return z


class Bottleneckblock(paddle.nn.Layer):
    def __init__(self, inplane, in_channel, out_channel, stride=1, start=False):
        super(Bottleneckblock, self).__init__()
        self.stride = stride
        self.start = start
        self.conv0 = Conv2D(in_channel, inplane, 1, stride=stride)
        self.conv1 = Conv2D(inplane, inplane, 3, stride=1, padding=1)
        self.conv2 = Conv2D(inplane, out_channel, 1, stride=1)
        self.conv3 = Conv2D(in_channel, out_channel, 1, stride=stride)
        self.bn0 = BatchNorm2D(inplane)
        self.bn1 = BatchNorm2D(inplane)
        self.bn2 = BatchNorm2D(out_channel)
        self.bn3 = BatchNorm2D(out_channel)

    def forward(self, inputs):
        y = inputs
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.start:
            y = self.conv3(y)
            y = self.bn3(y)
        z = F.relu(x + y)
        return z

# 定义卷积网络
class Res_CNN(paddle.nn.Layer):
    def __init__(self, num, level):
        super(Res_CNN, self).__init__()
        self.dict_dim = vocab["<pad>"]
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.channels = 1
        self.win_size = [3, self.hid_dim]
        self.batch_size = 32
        self.seq_len = 150
        self.embedding = Embedding(self.dict_dim + 1, self.emb_dim, sparse=False)
        self.hidden1 = paddle.nn.Conv2D(in_channels=1,  # 通道数
                                        out_channels=self.hid_dim,  # 卷积核个数
                                        kernel_size=self.win_size,  # 卷积核大小
                                        padding=[1, 1]
                                        )
        self.relu1 = paddle.nn.ReLU()
        self.hidden3 = paddle.nn.MaxPool2D(kernel_size=2,  # 池化核大小
                                           stride=2)  # 池化步长2
        self.hidden4 = paddle.nn.Linear(128 * 75, 512)

        self.conv0 = Conv2D(3, 64, 7, stride=2)
        self.bn = BatchNorm2D(64)
        self.pool1 = MaxPool2D(3, stride=2)
        if level == '50':
            self.layer0 = self.add_bottleneck_layer(num[0], 64, start=True)
            self.layer1 = self.add_bottleneck_layer(num[1], 128)
            self.layer2 = self.add_bottleneck_layer(num[2], 256)
            self.layer3 = self.add_bottleneck_layer(num[3], 512)
        else:
            self.layer0 = self.add_basic_layer(num[0], 64, start=True)
            self.layer1 = self.add_basic_layer(num[1], 128)
            self.layer2 = self.add_basic_layer(num[2], 256)
            self.layer3 = self.add_basic_layer(num[3], 512)
        self.pool2 = AdaptiveAvgPool2D(output_size=(1, 1))
        self.hidden5 = paddle.nn.Linear(512, 3)

    def add_basic_layer(self, num, inplane, start=False):
        layer = []
        if start:
            layer.append(Basicblock(inplane, inplane))
        else:
            layer.append(Basicblock(inplane // 2, inplane, stride=2))
        for i in range(num - 1):
            layer.append(Basicblock(inplane, inplane))
        return nn.Sequential(*layer)

    def add_bottleneck_layer(self, num, inplane, start=False):
        layer = []
        if start:
            layer.append(Bottleneckblock(inplane, inplane, inplane * 4, start=True))
        else:
            layer.append(Bottleneckblock(inplane, inplane * 2, inplane * 4, stride=2, start=True))
        for i in range(num - 1):
            layer.append(Bottleneckblock(inplane, inplane * 4, inplane * 4))
        return nn.Sequential(*layer)

    # 网络的前向计算过程
    def forward(self, inputs, input):

        # print('输入维度：', input.shape)
        x = self.embedding(input)
        batch_size = x.shape[0]  # Get the batch size dynamically
        x = paddle.reshape(x, [batch_size, 1, 150, 128])
        # x = paddle.reshape(x, [32, 1, 150, 128])
        x = self.hidden1(x)
        x = self.relu1(x)
        # print('第一层卷积输出维度：', x.shape)
        x = self.hidden3(x)
        # print('池化后输出维度：', x.shape)
        # 在输入全连接层时，需将特征图拉平会自动将数据拉平.
        x = paddle.reshape(x, shape=[batch_size, -1])
        # x = paddle.reshape(x, shape=[self.batch_size, -1])
        out1 = self.hidden4(x)

        x = self.conv0(inputs)
        x = self.bn(x)
        x = self.pool1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = paddle.squeeze(x)
        out = out1 + x
        out = self.hidden5(out)

        return out