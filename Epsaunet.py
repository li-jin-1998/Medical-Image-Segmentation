import torch
import torch.nn as nn
import math


class channel_attention(nn.Module):
	# 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
	def __init__(self, in_channel, ratio=4):
		# 继承父类初始化方法
		super(channel_attention, self).__init__()
		
		# 全局最大池化 [b,c,h,w]==>[b,c,1,1]
		self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
		# 全局平均池化 [b,c,h,w]==>[b,c,1,1]
		self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
		
		# 第一个全连接层, 通道数下降4倍
		self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
		# 第二个全连接层, 恢复通道数
		self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
		
		# relu激活函数
		self.relu = nn.ReLU()
		# sigmoid激活函数
		self.sigmoid = nn.Sigmoid()
	
	# 前向传播
	def forward(self, inputs):
		# 获取输入特征图的shape
		b, c, h, w = inputs.shape
		
		# 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
		max_pool = self.max_pool(inputs)
		# 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
		avg_pool = self.avg_pool(inputs)
		
		# 调整池化结果的维度 [b,c,1,1]==>[b,c]
		max_pool = max_pool.view([b, c])
		avg_pool = avg_pool.view([b, c])
		
		# 第一个全连接层下降通道数 [b,c]==>[b,c//4]
		x_maxpool = self.fc1(max_pool)
		x_avgpool = self.fc1(avg_pool)
		
		# 激活函数
		x_maxpool = self.relu(x_maxpool)
		x_avgpool = self.relu(x_avgpool)
		
		# 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
		x_maxpool = self.fc2(x_maxpool)
		x_avgpool = self.fc2(x_avgpool)
		
		# 将这两种池化结果相加 [b,c]==>[b,c]
		x = x_maxpool + x_avgpool
		# sigmoid函数权值归一化
		x = self.sigmoid(x)
		# 调整维度 [b,c]==>[b,c,1,1]
		x = x.view([b, c, 1, 1])
		# 输入特征图和通道权重相乘 [b,c,h,w]
		# outputs = inputs * x
		outputs = x
		return outputs
class SEWeightModule(nn.Module):

	def __init__(self, channels, reduction=16):
		super(SEWeightModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = self.avg_pool(x)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		weight = self.sigmoid(out)

		return weight
	
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
	"""standard convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
					 padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

	def __init__(self, inplans, planes, conv_kernels=[3,5,7,9], stride=1, conv_groups=[1, 4, 8, 16]):
		#groups 保证相同的计算量大小  不同的卷积核  空洞卷积应该可以省略这一步
		super(PSAModule, self).__init__()
		conv_groups = [1, 1, 1, 1]
		self.conv_1 = conv(inplans, planes//4, kernel_size=3, padding=3//2,dilation=1,
							stride=stride, groups=conv_groups[0])
		self.conv_2 = conv(inplans, planes//4, kernel_size=3, padding=5//2,dilation=2,
							stride=stride, groups=conv_groups[1])
		self.conv_3 = conv(inplans, planes//4, kernel_size=3, padding=7//2,dilation=3,
							stride=stride, groups=conv_groups[2])
		self.conv_4 = conv(inplans, planes//4, kernel_size=1, padding=0, dilation=1,
							stride=stride, groups=conv_groups[3])
		# self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
		#                    stride=stride, groups=conv_groups[0])
		# self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
		#                    stride=stride, groups=conv_groups[1])
		# self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
		#                    stride=stride, groups=conv_groups[2])
		# self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
		#                    stride=stride, groups=conv_groups[3])
		# self.se = SEWeightModule(planes // 4)
		self.se = channel_attention(planes // 4)
		self.split_channel = planes // 4
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		batch_size = x.shape[0]
		x1 = self.conv_1(x)
		x2 = self.conv_2(x)
		x3 = self.conv_3(x)
		x4 = self.conv_4(x)
		# print(x1.shape,x2.shape,x3.shape,x4.shape)
		feats = torch.cat((x1, x2, x3, x4), dim=1)
		feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

		x1_se = self.se(x1)
		x2_se = self.se(x2)
		x3_se = self.se(x3)
		x4_se = self.se(x4)
		# print(x1_se.shape, x2_se.shape, x3_se.shape, x4_se.shape)
		x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
		attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
		attention_vectors = self.softmax(attention_vectors)
		feats_weight = feats * attention_vectors
		for i in range(4):
			x_se_weight_fp = feats_weight[:, i, :, :]
			if i == 0:
				out = x_se_weight_fp
			else:
				out = torch.cat((x_se_weight_fp, out), 1)

		return out


class EPSABlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
				 conv_groups=[1, 4, 8, 16]):
		super(EPSABlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv(inplanes, planes,stride=stride)
		self.bn1 = norm_layer(planes)
		self.conv2 = PSAModule(planes, planes, stride=1, conv_kernels=conv_kernels, conv_groups=conv_groups)
		self.bn2 = norm_layer(planes)
		# self.conv3 = conv1x1(planes, planes * self.expansion)
		# self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		# out = self.relu(out)

		# out = self.conv3(out)
		# out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)
		# print(out.shape,identity.shape)
		out += identity
		out = self.relu(out)
		return out
import torch.nn.functional as F

from functools import partial
nonlinearity = partial(F.relu, inplace=True)
class EPSANet(nn.Module):
	def __init__(self,block, layers, num_classes=1000):
		super(EPSANet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layers(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layers(self, block, planes, num_blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, num_blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
class DecoderBlock(nn.Module):
	def __init__(self, in_channels, n_filters):
		super(DecoderBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
		self.norm1 = nn.BatchNorm2d(in_channels // 4)
		self.relu1 = nonlinearity

		self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
		self.norm2 = nn.BatchNorm2d(in_channels // 4)
		self.relu2 = nonlinearity

		self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
		self.norm3 = nn.BatchNorm2d(n_filters)
		self.relu3 = nonlinearity

	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.relu1(x)
		x = self.deconv2(x)
		x = self.norm2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu3(x)
		return x


class spatial_attention(nn.Module):
	# 初始化，卷积核大小为7*7
	def __init__(self, in_channel, dilation=1, kernel_size=7):
		# 继承父类初始化方法
		super(spatial_attention, self).__init__()
		# 空洞卷积
		self.conv_d = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, dilation=dilation,
		                        padding=dilation, bias=False)
		# 为了保持卷积前后的特征图shape相同，卷积时需要padding
		padding = kernel_size // 2
		# 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
		self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
		                      padding=padding, bias=False)
		# sigmoid函数
		self.sigmoid = nn.Sigmoid()
	
	# 前向传播
	def forward(self, inputs):
		# 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
		# 返回值是在某维度的最大值和对应的索引
		inputs = self.conv_d(inputs)
		
		x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
		
		# 在通道维度上平均池化 [b,1,h,w]
		x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
		# 池化后的结果在通道维度上堆叠 [b,2,h,w]
		x = torch.cat([x_maxpool, x_avgpool], dim=1)
		
		# 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
		x = self.conv(x)
		# 空间权重归一化
		x = self.sigmoid(x)
		# 输入特征图和空间权重相乘
		outputs = inputs * x
		
		return outputs
class Epsaunet(nn.Module):
	def __init__(self, num_classes=1, num_channels=3, pretrained=True):
		super(Epsaunet, self).__init__()
		
		super().__init__()
		filters = [64, 128, 256, 512]
		from torchvision import models
		# resnet = models.resnet34(pretrained=True)
		resnet = EPSANet(EPSABlock,[3, 4, 6, 3])
		self.firstconv = resnet.conv1
		self.firstbn = resnet.bn1
		self.firstrelu = resnet.relu
		self.firstmaxpool = resnet.maxpool
		self.encoder1 = resnet.layer1
		self.encoder2 = resnet.layer2
		self.encoder3 = resnet.layer3
		self.encoder4 = resnet.layer4
		# self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		# self.firstbn = nn.BatchNorm2d(64)
		# self.firstrelu = nn.ReLU(inplace=True)
		# self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		
		self.decoder4 = DecoderBlock(512, filters[2])
		self.decoder3 = DecoderBlock(filters[2], filters[1])
		self.decoder2 = DecoderBlock(filters[1], filters[0])
		self.decoder1 = DecoderBlock(filters[0], filters[0])
		
		self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
		self.finalrelu1 = nonlinearity
		self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.finalrelu2 = nonlinearity
		self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
		from CBAM import cbam
		self.cbam1 = spatial_attention(64, dilation=5)
		self.cbam2 = spatial_attention(128, dilation=3)
		self.cbam3 = spatial_attention(256, dilation=1)
	
	def forward(self, x):
		# Encoder
		x = self.firstconv(x)
		x = self.firstbn(x)
		x = self.firstrelu(x)
		x = self.firstmaxpool(x)
		# print(x.shape)
		e1 = self.encoder1(x)
		e2 = self.encoder2(e1)
		e3 = self.encoder3(e2)
		e4 = self.encoder4(e3)
		# Center
		# print(e1.shape,e2.shape,e3.shape,e4.shape)
		# Decoder
		d4 = self.decoder4(e4) + self.cbam3(e3)
		d3 = self.decoder3(d4) + self.cbam2(e2)
		d2 = self.decoder2(d3) + self.cbam1(e1)
		d1 = self.decoder1(d2)
		
		
		out = self.finaldeconv1(d1)
		out = self.finalrelu1(out)
		out = self.finalconv2(out)
		out = self.finalrelu2(out)
		out = self.finalconv3(out)
		
		return nn.Sigmoid()(out)
model = Epsaunet().to('cuda')
from torchsummary import summary
summary(model, (3, 512, 512))