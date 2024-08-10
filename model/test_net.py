import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from model.transformer import TransformerEncoder, TransformerDecoder
from model.base_function import init_net
from model.ffc import FFCResnetBlock, ConcatTupleLayer
from model.feature_fusion import FF
import numpy as np
import math

def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(patchsizes=[1, 2, 4, 8])  #old:[1,2,4,8]
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)

class Generator(nn.Module):
    def __init__(self, patchsizes):
        super().__init__()

        self.net = My_net(patchsizes=patchsizes)

    def forward(self, x, mask):
        out = self.net(x, mask)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def spectral_norm(module, mode=True):
    if mode:
        #一种用于对神经网络中的权重进行归一化的技术。
        #它通过在每次迭代中计算权重矩阵的最大奇异值，来保证网络中的权重不会过大，从而提高网络的稳定性和泛化能力
        return nn.utils.spectral_norm(module)

    return module

# compute the l2 distance between feature patches
def euclidean_matrix(s):
    outk = []

    for i in range(s):
        for k in range(s):

            out = []
            for x in range(s):
                row = []
                for y in range(s):
                    cord_x = i
                    cord_y = k
                    dis_x = abs(x - cord_x)
                    dis_y = abs(y - cord_y)
                    #dis_add = -(dis_x + dis_y)
                    dis_add = -math.sqrt(dis_x**2 + dis_y**2)
                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    return torch.from_numpy(out)     #用于将numpy数组转换为PyTorch张量

class conv_block_my(nn.Module):
    def __init__(self, in_channels, out_channels, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1):
        """

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        """
        super(conv_block_my, self).__init__()

        nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, dilation=1)
        #dilation参数在nn.Conv2d中用于控制卷积核的空洞大小，
        #当dilation=1时，卷积核中的每个元素之间没有空洞；当dilation>1时，卷积核中的元素之间有间隔，即空洞，间隔大小为dilation-1。

        self.conv_block_my = nn.Sequential(
            nn_Conv2d(in_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn_Conv2d(out_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """
        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, in_channels,  height, width]
            Input to the convolutional block.

        Returns:
        --------
        outputs: another 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the convolutional block.

        """
        outputs = self.conv_block_my(inputs)
        return outputs

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)  # std:计算标准差
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# -----------------------------------------------
#                Gated ARMAConvBlock
# -----------------------------------------------
class GatedARMAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1,
                 pad_type='zero', activation='lrelu', norm='none'):
        super(GatedARMAConv2d, self).__init__()

        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv2d = conv_block_my(in_channels, out_channels, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.mask_conv2d = conv_block_my(in_channels, out_channels, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1,
                 activation='lrelu', norm='none', scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedARMAConv2d(in_channels, out_channels, w_stride=1, w_dilation=1, a_stride=1,
                                            a_dilation=1, activation=activation, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.gated_conv2d(x)
        return x

class FFCBlock(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of output/input channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 padding,
                 ratio_gin=0.75,
                 ratio_gout=0.75,
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 ):
        super().__init__()
        if activation == 'linear':
            self.activation = nn.Identity
        else:
            self.activation = nn.ReLU
        self.padding = padding
        self.kernel_size = kernel_size
        self.ffc_block = FFCResnetBlock(dim=dim,
                                        padding_type='reflect',
                                        norm_layer=nn.SyncBatchNorm,
                                        activation_layer=self.activation,
                                        dilation=1,
                                        ratio_gin=ratio_gin,
                                        ratio_gout=ratio_gout)

        self.concat_layer = ConcatTupleLayer()

    def forward(self, gen_ft, mask, fname=None):
        x = gen_ft.float()

        x_l, x_g = x[:, :-self.ffc_block.conv1.ffc.global_in_num], x[:, -self.ffc_block.conv1.ffc.global_in_num:]
        id_l, id_g = x_l, x_g

        x_l, x_g = self.ffc_block((x_l, x_g), fname=fname)
        x_l, x_g = id_l + x_l, id_g + x_g
        x = self.concat_layer((x_l, x_g))

        return x + gen_ft.float()


class FFCSkipLayer(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of input/output channels.
                 kernel_size=3,  # Convolution kernel size.
                 ratio_gin=0.75,
                 ratio_gout=0.75,
                 ):
        super().__init__()
        self.padding = kernel_size // 2

        self.ffc_act = FFCBlock(dim=dim, kernel_size=kernel_size, activation=nn.ReLU,
                                padding=self.padding, ratio_gin=ratio_gin, ratio_gout=ratio_gout)

    def forward(self, gen_ft, mask, fname=None):
        x = self.ffc_act(gen_ft, mask, fname=fname)
        return x

class My_net(nn.Module):
    def __init__(self, patchsizes, in_channels=4, out_channels=3, factor=1, max_ngf=256):
        """
        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.


        """
        super(My_net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2)
        self.Conv1 = GatedARMAConv2d(in_channels, 64 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)

        self.Conv2 = GatedARMAConv2d(64 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.Conv3 = GatedARMAConv2d(128 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.Conv4 = GatedARMAConv2d(128 // factor, 256 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.Conv5 = GatedARMAConv2d(256 // factor, 256 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)



        for i, pz in enumerate(patchsizes):
            length = 64 // pz
            dis = euclidean_matrix(length)
            dis = dis.view(length * length, length, length).float()
            block = TransformerEncoder(patchsizes=pz, num_hidden=max_ngf, dis=dis)
            #setattr：用于设置对象的属性值 语法：setattr(object, name, value)
            #object：必选参数，表示要设置属性值的对象l;  name：必选参数，表示要设置的属性名称；  value：必选参数，表示要设置的属性值。
            setattr(self, 'transE'+str(i+1), block)
            block = TransformerDecoder(patchsizes=pz, num_hidden=max_ngf, dis=dis)
            setattr(self, 'transD'+str(i+1), block)


        #self.Up5 = TransposeGatedConv2d(256 // factor, 256 // factor)
        self.Up_conv5 = GatedARMAConv2d(256 // factor, 256 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                        a_dilation=1)
        #self.Up_ffc5 = FFCSkipLayer(256)
        #self.Up4 = TransposeGatedConv2d(256 // factor, 128 // factor)
        self.Up_conv5_ = GatedARMAConv2d(256 // factor, 256 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                         a_dilation=1)


        self.Ch_conv4 = GatedARMAConv2d(256 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                        a_dilation=1)
        self.Up_conv4 = GatedARMAConv2d(128 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                        a_dilation=1)
        self.Up_ffc4 = FFCSkipLayer(128)
        self.Up_ff4 = FF(128, 128)


        self.Up3 = TransposeGatedConv2d(128 // factor, 128 // factor)


        self.Up_conv3_ = GatedARMAConv2d(128 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                         a_dilation=1)
        self.Up_conv3 = GatedARMAConv2d(128 // factor, 128 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                        a_dilation=1)
        self.Up_ffc3 = FFCSkipLayer(128)
        self.Up_ff3 = FF(128, 128)


        self.Up2 = TransposeGatedConv2d(128 // factor, 64 // factor)


        self.Up_conv2_ = GatedARMAConv2d(64 // factor, 64 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                         a_dilation=1)
        self.Up_conv2 = GatedARMAConv2d(64 // factor, 64 // factor, w_stride=1, w_dilation=1, a_stride=1,
                                        a_dilation=1)
        self.Up_ffc2 = FFCSkipLayer(64)
        self.Up_ff2 = FF(64, 64)


        #self.Up_ffc1 = FFCSkipLayer(64)
        self.Up_conv1 = GatedARMAConv2d(64 // factor, 32 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.Up_conv0 = GatedARMAConv2d(32 // factor, 32 // factor, w_stride=1, w_dilation=1, a_stride=1, a_dilation=1)
        self.Conv_1x1 = nn.Conv2d(32 // factor, out_channels, 1)


    def forward(self, in1, in2):
        """

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, in_channels, height, width]


        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]


        """

        # encoding path
        noise = torch.normal(mean=torch.zeros_like(in1), std=torch.ones_like(in1) * (1. / 256.))
        x = in1 + noise
        x = torch.cat((x, in2), dim=1)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        #x4 = self.Maxpool(x3)
        x4 = self.Conv4(x3)

        #x5 = self.Maxpool(x4)
        x5 = self.Conv5(x4)

        feature = self.transE1(x5)
        feature = self.transE2(feature)
        feature = self.transE3(feature)
        feature = self.transE4(feature)

        #old: feature = self.transD4(feature, feature, feature)
        feature = self.transD3(feature, feature, feature)
        feature = self.transD2(feature, feature, feature)
        feature = self.transD1(feature, feature, feature)

        # decoding + concat path
        #d5 = self.Up5(feature)

        d5 = self.Up_conv5(feature)
        d5 = x4 + d5
        #d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5_(d5)
        #f5 = self.Up_ffc5(d5, in2)

        #d4 = self.Up4(f5)
        d4 = self.Ch_conv4(d5)
        res4 = d4
        d4 = x3 + d4
        #d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        mask4 = F.interpolate(in2, size=d4.shape[2:], )
        f4 = self.Up_ffc4(d4, mask4)
        ff4 = self.Up_ff4(res4, x3, f4)
        d4 = res4 + ff4

        d3 = self.Up3(d4)

        d3 = self.Up_conv3_(d3)
        res3 = d3
        d3 = x2 + d3
        #d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        mask3 = F.interpolate(in2, size=d3.shape[2:], )
        f3 = self.Up_ffc3(d3, mask3)
        ff3 = self.Up_ff3(res3, x2, f3)
        d3 = res3 + ff3

        d2 = self.Up2(d3)

        d2 = self.Up_conv2_(d2)
        res2 = d2
        d2 = x1 + d2
        #d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        f2 = self.Up_ffc2(d2, in2)
        ff2 = self.Up_ff2(res2, x1, f2)
        d2 = res2 + ff2

        #f1 = self.Up_ffc1(d2, in2)

        d1 = self.Up_conv1(d2)
        d1 = self.Up_conv0(d1)
        d1 = self.Conv_1x1(d1)

        return d1