import numpy as np
from tqdm import tqdm
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot


def getArgs():
	parse = argparse.ArgumentParser()
	parse.add_argument('--deepsupervision', default=0)
	parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
	parse.add_argument("--epoch", type=int, default=203)
	parse.add_argument('--arch', '-a', metavar='ARCH', default='resunet_cbam2',
	                   help='unet/cenet/resunet/r2unet/m2unet/attention_unet/epsaunet2/resunet_cbam2')
	parse.add_argument("--batch_size", type=int, default=16)
	parse.add_argument('--dataset', default='E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/', help='path')
	parse.add_argument("--ckp", type=str, help="the path of model weight file")
	parse.add_argument("--log_dir", default='result/log', help="log dir")
	parse.add_argument("--threshold", type=float, default=None)
	args = parse.parse_args(args=[])
	return args

from getModel import getModel

def getDataset(args):
	train_dataset = Dataset(r"train", path=args.dataset, transform=x_transforms, target_transform=y_transforms)
	train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
	val_dataset = Dataset(r"val", path=args.dataset, transform=x_transforms, target_transform=y_transforms)
	val_dataloaders = DataLoader(val_dataset, batch_size=16)
	test_dataset = Dataset(r"test", path=args.dataset, transform=x_transforms, target_transform=y_transforms)
	test_dataloaders = DataLoader(test_dataset, batch_size=16)
	return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_dice, val_dataloaders):
	print('**************************')
	print('Val')
	print('**************************')
	model = model.eval()
	with torch.no_grad():
		i = 0  # 验证集中第i张图
		dices = []
		num = len(val_dataloaders)  # 验证集图片的总数
		for x, l, pic, mask in tqdm(val_dataloaders):
			x = x.to(device)
			y = model(x)
			if args.deepsupervision:
				img_y = torch.squeeze(y[-1]).cpu().numpy()
			else:
				img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
			l = l.numpy()
			l = np.squeeze(l)
			img_y[img_y > 0.5] = 1
			img_y[img_y <= 0.5] = 0
			y_true_f = l.flatten()
			y_pred_f = img_y.flatten()
			intersection = sum(y_true_f * y_pred_f)
			smooth = 1
			dice = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
			# print(i,dice)
			dices.append(dice)
			if i < num: i += 1
		aver_dice = np.mean(dices)
		print('aver_dice=%f' % (aver_dice))
		print('best_dice=%f' % (best_dice))
		if aver_dice > best_dice:
			print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
			best_dice = aver_dice
			print('===========>save best model!')
			torch.save(model.state_dict(),
			           r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth')
		torch.save(model.state_dict(),
		           r'./saved_model/latest' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
			           args.epoch) + '.pth')
		return best_dice, aver_dice

def val_model(model,best_dice):
	print('**************************')
	print('Val model')
	print('**************************')
	import os,shutil
	result_path = './result/'
	if os.path.exists(result_path):
		shutil.rmtree(result_path)
	if not os.path.exists(result_path):
		os.mkdir(result_path)
		os.mkdir(result_path + 'predict')
		os.mkdir(result_path + 'image')
		os.mkdir(result_path + 'seg')
	model = model.eval()
	with torch.no_grad():
		for path in tqdm(os.listdir('E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/test_one/image/')):
			Volume_test(model, path,verbose=False)
		from Compute_metrics import dice, volumeMetrics
		import os
		import SimpleITK as sitk
		origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/seg/'
		label_paths = os.listdir(origin_path)
		dice1 = []
		for path in label_paths:
			# print(path)
			label = sitk.ReadImage(origin_path + path, sitk.sitkUInt8)
			predict = sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/predict/' + path,
			                         sitk.sitkUInt8)
			dice1.append(volumeMetrics(label, predict))
			# print('Pre', volumeMetrics(label, predict))
			# print(np.mean(dice1), np.std(dice1))
		aver_dice = np.mean(dice1)
		print('aver_dice=%f' % (aver_dice))
		print('best_dice=%f' % (best_dice))
		if aver_dice > best_dice:
			print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
			best_dice = aver_dice
			print('===========>save best model!')
			torch.save(model.state_dict(),
			           r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth')
		torch.save(model.state_dict(),
		           r'./saved_model/latest' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
			           args.epoch) + '.pth')
		return best_dice, aver_dice


def Volume_test(model, path, verbose=True):
	import SimpleITK as sitk
	origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/test_one/image/'
	if verbose:
		print(origin_path + path)
	origin_image = sitk.ReadImage(origin_path + path, sitk.sitkFloat32)
	array = sitk.GetArrayFromImage(origin_image)
	origin_seg_path = origin_path.replace('image', 'seg')
	path2 = path
	seg_array = sitk.GetArrayFromImage(sitk.ReadImage(origin_seg_path + path2, sitk.sitkUInt8))
	# array = window_transform(array, 300, 100, normal=False)
	array = array * 255
	count = 0
	import shutil
	preprocess_path = './temp/'
	if os.path.exists(preprocess_path):
		shutil.rmtree(preprocess_path)
	os.makedirs(preprocess_path)
	os.makedirs(preprocess_path + 'image')
	os.makedirs(preprocess_path + 'seg')
	batchsize = 8
	for i in range(array.shape[0]):
		image = array[i, :, :]  # *seg_array[i,:,:]#提取mask部分
		label = seg_array[i, :, :] * 255
		cv2.imwrite('./temp/image/' + str(count).rjust(3, '0') + '.png', image)
		cv2.imwrite('./temp/seg/' + str(count).rjust(3, '0') + '.png', label)
		count += 1
	test_dataset = Dataset(r"test1", path='./temp/', transform=x_transforms, target_transform=y_transforms)
	test_dataloaders = DataLoader(test_dataset, batch_size=batchsize)
	dices = []
	num = len(test_dataloaders)  # 验证集图片的总数
	predict = np.zeros(array.shape)
	img2 = np.zeros(array.shape)
	id = 0
	i = 0
	model = model.eval()
	with torch.no_grad():
		for x, l, pic, mask in test_dataloaders:
			k = x[:, 1, :, :]
			x = x.to(device)
			y = model(x)
			if args.deepsupervision:
				img_y = torch.squeeze(y[-1]).cpu().numpy()
			else:
				img_y = torch.squeeze(y).cpu().detach().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
			l = l.numpy()
			l = np.squeeze(l)
			img_y[img_y > 0.5] = 1
			img_y[img_y <= 0.5] = 0
			predict[id * batchsize:(id + 1) * batchsize, :, :] = img_y
			y_true_f = l.flatten()
			y_pred_f = img_y.flatten()
			intersection = sum(y_true_f * y_pred_f)
			smooth = 1
			dice = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
			if verbose:
				print(i, dice)
			dices.append(dice)
			k = k.numpy()
			k = np.squeeze(k)
			if x.shape[0] != 8:
				img2[id * batchsize:, :, :] = k
			else:
				img2[id * batchsize:(id + 1) * batchsize, :, :] = k
			id = id + 1
			if i < num: i += 1
	predict = sitk.GetImageFromArray(predict)
	predict.SetDirection(origin_image.GetDirection())
	predict.SetOrigin([0, 0, 0])
	predict.SetSpacing([1, 1, 1])
	predict = sitk.Cast(predict, sitk.sitkUInt8)
	sitk.WriteImage(predict, './result/predict/' + path)
	shutil.copy(origin_path + path, './result/image/' + path)
	shutil.copy(origin_seg_path + path2, './result/seg/' + path)
def train(model, criterion0, criterion, optimizer, train_dataloader, val_dataloader, args, scheduler=None):
	print('**************************')
	print('Train')
	print('**************************')
	best_dice, aver_dice = 0, 0
	num_epochs = args.epoch
	threshold = args.threshold
	loss_list = []
	for epoch in range(num_epochs):
		model = model.train()
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		epoch_loss = 0
		for x, y, _, mask in tqdm(train_dataloader):
			inputs = x.to(device)
			labels = y.to(device)
			optimizer.zero_grad()
			if args.deepsupervision:
				outputs = model(inputs)
				loss = 0
				for output in outputs:
					a = 0.5
					loss += (1 - a) * criterion(output, labels) + a * criterion0(output, labels)
				loss /= len(outputs)
			else:
				output = model(inputs)
				a = 0.5
				loss = (1 - a) * criterion(output, labels) + a * criterion0(output, labels)
			if threshold != None:
				if loss > threshold:
					loss.backward()
					optimizer.step()
					epoch_loss += loss.item()
			else:
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
		if scheduler is not None:
			scheduler.step()
		loss_list.append(epoch_loss)
		best_dice, aver_dice = val_model(model, best_dice)
		# best_dice, aver_dice = val(model, best_dice,val_dataloader)
		print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
	# loss_plot(args, loss_list)
	return model

if __name__ == "__main__":
	x_transforms = transforms.Compose([
		transforms.ToTensor(),  # -> [0,1]
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
	])
	
	# mask只需要转换为tensor
	y_transforms = transforms.ToTensor()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	args = getArgs()
	print('**************************')
	print('models:%s\nepoch:%s\nbatch size:%s' % \
	      (args.arch, args.epoch, args.batch_size))
	print('**************************')
	model = getModel(args)
	weight = r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	# weight=r'./saved_model/latest' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	model.load_state_dict(torch.load(weight, map_location='cuda'))
	train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
	# loss function BCE  二进制交叉熵   Sigmoid +BCELoss.
	from dice_loss import SoftDiceLoss
	criterion0 = torch.nn.BCELoss()
	criterion = SoftDiceLoss()
	
	# 优化器 Adam  SGD  lr_scheduler学习率调整
	optimizer = optim.Adam(model.parameters(), lr=0.0004)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.99, verbose=True)
	if 'train' in args.action:
		train(model, criterion0, criterion, optimizer, train_dataloaders, val_dataloaders, args, scheduler=scheduler)
		# torch.save(model.state_dict(), r'./saved_model/final' + str(args.arch) + '_' + str(args.batch_size) +
		#            '_' + str(args.epoch) + '.pth')
