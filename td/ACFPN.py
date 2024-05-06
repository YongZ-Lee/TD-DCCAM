from collections import OrderedDict
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torch.nn.parameter import Parameter
import math

from detectron2.modeling import Backbone
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool

from .c2f_DLKA import C2f_DLKA, C2f, DeformConv, SPPF_LSKA, SPPF_DLSKA, autopad


from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 实现了CxAM和CnAM模块
class CxAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CxAM, self).__init__()
        self.key_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.query_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        # u = x.clone()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C'

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C' x N

        R = torch.bmm(proj_query, proj_key).view(m_batchsize, width * height, width, height)  # B x N x W x H
        # 先进行全局平均池化, 此时 R 的shape为 B x N x 1 x 1, 再进行view, R 的shape为 B x 1 x W x H
        attention_R = self.sigmoid(self.avg(R).view(m_batchsize, -1, width, height))  # B x 1 x W x H

        proj_value = self.value_conv(x)

        out = proj_value * attention_R  # B x W x H

        return out

class CnAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CnAM, self).__init__()
        # 原文中对应的P, Z, S
        self.Z_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.P_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    # CnAM使用了FPN中的F5和CEM输出的特征图F
    def forward(self, F5, F):
        m_batchsize, C, width, height = F5.size()

        proj_query = self.P_conv(F5).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C''

        proj_key = self.Z_conv(F5).view(m_batchsize, -1, width * height)  # B x C'' x N

        S = torch.bmm(proj_query, proj_key).view(m_batchsize, width * height, width, height)  # B x N x W x H
        attention_S = self.sigmoid(self.avg(S).view(m_batchsize, -1, width, height))  # B x 1 x W x H

        proj_value = self.value_conv(F)

        out = proj_value * attention_S  # B x W x H

        return out

class DLKA(nn.Module):
    def __init__(self, input_num, output_num, n=1):
        super().__init__()

        self.spf_dlska = SPPF_DLSKA(input_num, output_num)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter( layer_scale_init_value * torch.ones( (256) ), requires_grad=True )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):

        att = self.spf_dlska(x)

        x = self.drop( self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * att)

        return x

class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)

        self.dilaconv = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)

        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

        # weight_init.c2_xavier_fill(self.conv1x1)
        # weight_init.c2_xavier_fill(self.dilaconv)

    def forward(self, x):
        # print('---x1:', x.dtype)
        x = self.ConvGN(self.conv1x1(x))
        # print('---x2:', x.dtype)
        x = self.relu1(x)

        # print('---x3:', x.dtype)
        x = self.dilaconv(x)

        # print('---x4:', x.dtype)
        x = self.relu2(x)
        # print('---x5:', x.dtype)
        x = self.drop(x)
        # print('---x6:', x.dtype)
        return x

class DenseAPP(nn.Module):
    def __init__(self, num_channels=2048):
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2,
                                rate=3,
                                drop_out=self.drop_out)

        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)

        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)

        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)

        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)

        # self.conv1x1 = nn.Conv2d(in_channels=5 * self.channels2, out_channels=256, kernel_size=1)
        # self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.conv1x1 = nn.Conv2d(in_channels=6 * self.channels2, out_channels=256, kernel_size=1)
        # self.conv1x1 = nn.Conv2d(in_channels=3328, out_channels=256, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=256)

        self.sda = DLKA(768, 256, n=3)



    def forward(self, feature, feature2):

        # print("-------feature:", feature.size())
        # sim = self.simam(feature)
        # sim = self.ema(feature2)
        # sim = self.ca(feature)
        # sim = self.soc(feature2)
        # sim = self.eca(feature)
        # sim = self.gam(feature)

        # sim = self.mda(feature2)
        sim = self.sda(feature2)


        # sim = self.scc(feature)
        # print("-------sim:", sim.size())

        # print('---feature:', feature.dtype)
        aspp3 = self.aspp3(feature)
        # feature = torch.concat((aspp3, feature), dim=1)
        feature = torch.cat((aspp3, feature), dim=1)
        aspp6 = self.aspp6(feature)
        # feature = torch.concat((aspp6, feature), dim=1)
        feature = torch.cat((aspp6, feature), dim=1)
        aspp12 = self.aspp12(feature)
        # feature = torch.concat((aspp12, feature), dim=1)
        feature = torch.cat((aspp12, feature), dim=1)
        aspp18 = self.aspp18(feature)
        # feature = torch.concat((aspp18, feature), dim=1)
        feature = torch.cat((aspp18, feature), dim=1)
        aspp24 = self.aspp24(feature)

        # x = torch.concat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        x = torch.cat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        # print("-------x:", x.size())

        x_t = torch.cat((x, sim), dim=1)
        # print("-------x_t:", x_t.size())

        # out = self.ConvGN(self.conv1x1(x))
        out = self.ConvGN(self.conv1x1(x_t))
        # print("-------out:", out.size())
        # print('---out:', out.dtype)
        return out

class AC_FPN(Backbone):
    _fuse_type: torch.jit.Final[str]

    def __init__(
            self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        super(AC_FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        # self.dense = DenseAPP(num_channels=in_channels_list[-1])
        self.conv_1x1_5 = nn.Conv2d(768, 2048, 1)
        self.dense = DenseAPP(num_channels=2048)

        # --------增加AM模块，若不想使用，可直接注释掉--------#
        # self.conv_1x1_6 = nn.Conv2d(768, 256, 1)
        self.CxAM = CxAM(in_channels=256, out_channels=256)
        self.CnAM = CnAM(in_channels=256, out_channels=256)
        # -------------------------------------------------#

        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        # for in_channels in in_channels_per_feature:
        for idx, in_channels in enumerate(in_channels_per_feature):
            if in_channels == 0:
                continue
            stage = int(math.log2(strides[idx]))
            if stage == 5:
                inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
                layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            elif stage == 4:
                inner_block_module = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                )
                layer_block_module = nn.Sequential(
                    C2f(256, 256, 3, True),
                )
            else:
                inner_block_module = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                )
                layer_block_module = nn.Sequential(
                    C2f(256, 256, 3, True),
                )
            # inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            # layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            # print("((-----m:", m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            # weight_init.c2_xavier_fill(lateral_conv)
            # weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)


        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}

        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        # print('-----self._out_features: ', self._out_features)
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
                # print(f"********{i}", module)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
                # print(f"********{i}", module)
            i += 1
        return out

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []

        c2 = bottom_up_features[self.in_features[0]]
        c3 = bottom_up_features[self.in_features[1]]
        c4 = bottom_up_features[self.in_features[2]]
        c5 = bottom_up_features[self.in_features[3]]

        x_t = [c2, c3, c4, c5]


        # 将C5送入DenseAPP中获得上下文信息
        c5_t = self.conv_1x1_5(x_t[-1])
        dense = self.dense(c5_t, c5)
        dense = dense.to(c5_t.dtype)

        # 将resnet layer4的channel调整到指定的out_channels
        last_inner = self.get_result_from_inner_blocks(x_t[-1], -1)

        # 将dense送入cxam模块和cnam模块，不想使用AM模块注释下面三行即可
        cxam = self.CxAM(dense)
        cnam = self.CnAM(dense, last_inner)
        result = cxam + cnam

        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵

        # 不使用AM模块
        # P5 = dense + self.get_result_from_layer_blocks(last_inner, -1)

        # 使用AM模块
        P5 = result + self.get_result_from_layer_blocks(last_inner, -1)

        results.append(P5)

        for idx in range(len(x_t) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x_t[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down

            last_inner = self.get_result_from_layer_blocks(last_inner, idx)

            results.insert(0, last_inner)

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

