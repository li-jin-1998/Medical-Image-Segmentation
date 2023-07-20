import numpy as np
from tqdm import tqdm
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet, resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from torchvision.models import vgg16
import torch.nn.functional as F


def getArgs():
	parse = argparse.ArgumentParser()
	parse.add_argument('--deepsupervision', default=0)
	parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
	parse.add_argument("--epoch", type=int, default=203)
	parse.add_argument('--arch', '-a', metavar='ARCH', default='m2unet',
	                   help='unet/cenet/resunet/r2unet/m2unet')
	parse.add_argument("--batch_size", type=int, default=16)
	parse.add_argument('--dataset', default='E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/', help='path')
	parse.add_argument("--ckp", type=str, help="the path of model weight file")
	parse.add_argument("--log_dir", default='result/log', help="log dir")
	parse.add_argument("--threshold", type=float, default=None)
	# args = parse.parse_args()
	args = parse.parse_args(args=[])
	return args


def getLog(args):
	dirname = os.path.join(args.log_dir, args.arch, str(args.batch_size), str(args.epoch))
	filename = dirname + '/log.log'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	logging.basicConfig(
		filename=filename,
		level=logging.DEBUG,
		format='%(asctime)s:%(levelname)s:%(message)s'
	)
	return logging


def getModel(args):
	if args.arch == 'cenet':
		from cenet import CE_Net_
		from u2net import u2net_full, u2net_lite
		from r2unet import R2U_Net
		model = CE_Net_().to(device)
	if args.arch == 'unet':
		from UNet import Unet
		model = Unet().to(device)
	if args.arch == 'resunet':
		from UNet import resnet34_unet
		model = resnet34_unet().to(device)
	if args.arch == 'm2unet':
		from m2unet import CE_Net_
		model = CE_Net_().to(device)
	# from torchsummary import summary
	# summary(model, (3, 512, 512))
	# exit(0)
	return model
	# from torchsummary import summary
	# summary(model, (3, 512, 512))
	# exit(0)
	from ptflops import get_model_complexity_info
	# model = torchvision.models.alexnet(pretrained=False)
	# flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
	# print('flops: ', flops, 'params: ', params)


def getDataset(args):
	train_dataset = Dataset(r"train", path=args.dataset, transform=x_transforms, target_transform=y_transforms)
	train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
	val_dataset = Dataset(r"val", path=args.dataset, transform=x_transforms, target_transform=y_transforms)
	val_dataloaders = DataLoader(val_dataset, batch_size=args.batch_size)
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
			# print(np.min(l),np.max(l))
			# print(np.min(img_y),np.max(img_y))
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


loss1 = 0


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
			# print(type(inputs))
			# zero the parameter gradients
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
				# output=torch.stack(output)
				# output[output>=0.5]=1
				# output[output <0.5] = 0
				# print(type(output))
				a = 0.5
				# loss=criterion(output[0], labels)
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
		# print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
		# loss1 = loss
		loss_list.append(epoch_loss)
		best_dice, aver_dice = val(model, best_dice, val_dataloader)
		print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
	
	# loss_plot(args, loss_list)
	# metrics_plot(args, 'iou&dice',iou_list, dice_list)
	# metrics_plot(args,'hd',hd_list)
	return model


def K_test(test_dataloaders, save_predict=None):
	print('**************************')
	print('Test')
	print('**************************')
	if save_predict == True:
		dir = os.path.join(r'./saved_predict', str(args.arch))
		if not os.path.exists(dir):
			os.makedirs(dir)
		else:
			print('dir already exist!')
	print(r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth')
	model.load_state_dict(
		torch.load(r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth',
		           map_location='cpu'))  # 载入训练好的模型
	model.eval()
	
	with torch.no_grad():
		i = 0  # 验证集中第i张图
		miou_total = 0
		hd_total = 0
		dice_total = 0
		num = len(test_dataloaders)  # 验证集图片的总数
		totalThread = -1  # 需要创建的线程数，可以控制线程的数量
		if totalThread == 1:
			for pic, _, pic_path, mask_path in tqdm(test_dataloaders):
				pic = pic.to(device)
				predict = model(pic)
				if args.deepsupervision:
					predict = torch.squeeze(predict[-1]).cpu().numpy()
				else:
					predict = torch.squeeze(
						predict).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
				# img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
				predict = cv2.resize(predict, (512, 512), interpolation=cv2.INTER_CUBIC)
				# print(predict.shape)
				iou = get_iou(mask_path[0], predict)
				miou_total += iou  # 获取当前预测图的miou，并加到总miou中
				hd_total += get_hd(mask_path[0], predict)
				dice = get_dice(mask_path[0], predict)
				dice_total += dice
				
				fig = plt.figure()
				plt.title('dice:{}'.format(dice))
				ax1 = fig.add_subplot(1, 3, 1)
				ax1.set_title('input')
				plt.imshow(Image.open(pic_path[0]), cmap='Greys_r')
				# print(pic_path[0])
				ax2 = fig.add_subplot(1, 3, 2)
				ax2.set_title('predict')
				plt.imshow(predict, cmap='Greys_r')
				ax3 = fig.add_subplot(1, 3, 3)
				ax3.set_title('mask')
				plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
				# print(mask_path[0])
				
				if save_predict == True:
					# if args.dataset == 'driveEye':
					#     saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
					#     saved_predict = '.' + saved_predict.split('.')[1] + '.tif'
					#     plt.savefig(saved_predict)
					# else:
					plt.savefig(dir + '/' + mask_path[0].split('\\')[-1])
				
				import cv2 as cv
				predict = predict.astype('uint8')
				# print(np.sum(predict))
				predict[predict > 0.5] = 255
				# r,t=cv.threshold(predict,127,255,0)
				cv.imwrite('./result/result_label/' + mask_path[0].split('\\')[-1],
				           predict)  #########################结果在这里要改一下名字
				if i < num: i += 1  # 处理验证集下一张图
			# plt.show()
			print('Miou=%f,aver_hd=%f,aver_dice=%f' % (miou_total / num, hd_total / num, dice_total / num))
		else:
			dices = []
			num = len(test_dataloaders)  # 验证集图片的总数
			for x, l, pic, mask in test_dataloaders:
				k = x[:, 1, :, :]
				# print(k.shape)
				x = x.to(device)
				# print(x.shape)
				y = model(x)
				if args.deepsupervision:
					img_y = torch.squeeze(y[-1]).cpu().numpy()
				else:
					img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
				l = l.numpy()
				l = np.squeeze(l)
				# print(l.shape)
				# print(img_y.shape)
				
				img_y[img_y > 0.5] = 1
				img_y[img_y <= 0.5] = 0
				# print(np.min(l),np.max(l))
				# print(np.min(img_y),np.max(img_y))
				y_true_f = l.flatten()
				y_pred_f = img_y.flatten()
				intersection = sum(y_true_f * y_pred_f)
				smooth = 1
				dice = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
				print(i, dice)
				dices.append(dice)
				import SimpleITK as sitk
				l[l > 0.5] = 2
				img_y = img_y + l
				label = sitk.GetImageFromArray(img_y)
				sitk.WriteImage(label, './result/label' + str(i) + '.nii.gz')
				# print(k.shape)
				k = k.numpy()
				k = np.squeeze(k)
				img = sitk.GetImageFromArray(k)
				sitk.WriteImage(img, './result/img' + str(i) + '.nii.gz')
				if i < num: i += 1
			aver_dice = np.mean(dices)
			print('aver_dice=%f' % (aver_dice))


def window_transform(ct_array, windowWidth, windowCenter, normal=True):
	minWindow = float(windowCenter) - 0.5 * float(windowWidth)
	newimg = (ct_array - minWindow) / float(windowWidth)
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	if not normal:
		newimg = (newimg * 255).astype('float32')
	return newimg


def Volume_test(path):
	import SimpleITK as sitk
	origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/test_one/image/'
	print(origin_path + path)
	# origin_path = 'E:/lijin/radiomics/resample/image/'
	origin_image = sitk.ReadImage(origin_path + path, sitk.sitkFloat32)
	array = sitk.GetArrayFromImage(origin_image)
	origin_seg_path = origin_path.replace('image', 'seg')
	path2 = path
	# path2=path.replace('pancreas_','label00')
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
	# print(array.shape[0])
	batchsize = 8
	# for j in range(20, 1, -1):
	# 	if array.shape[0] % j == 0:
	# 		batchsize = j
	# 		break
	print(batchsize)
	for i in range(array.shape[0]):
		image = array[i, :, :]  # *seg_array[i,:,:]#提取mask部分
		label = seg_array[i, :, :] * 255
		# seg_image = np.rot90(np.transpose(seg_image, (1,0)))
		# origin_image = np.rot90(np.transpose(origin_image, (1,0)))
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
	for x, l, pic, mask in test_dataloaders:
		# print(pic,mask)
		k = x[:, 1, :, :]
		# print(k.shape)
		x = x.to(device)
		# print(x.shape)
		y = model(x)
		if args.deepsupervision:
			img_y = torch.squeeze(y[-1]).cpu().numpy()
		else:
			img_y = torch.squeeze(y).cpu().detach().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
		l = l.numpy()
		l = np.squeeze(l)
		# print(l.shape)
		# print(img_y.shape)
		
		img_y[img_y > 0.5] = 1
		img_y[img_y <= 0.5] = 0
		# print(img_y.shape)
		predict[id * batchsize:(id + 1) * batchsize, :, :] = img_y
		# print(np.min(l),np.max(l))
		# print(np.min(img_y),np.max(img_y))
		y_true_f = l.flatten()
		y_pred_f = img_y.flatten()
		intersection = sum(y_true_f * y_pred_f)
		smooth = 1
		dice = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
		print(i, dice)
		dices.append(dice)
		import SimpleITK as sitk
		# l[l > 0.5] = 2
		# img_y = img_y + l
		# label = sitk.GetImageFromArray(img_y)
		# sitk.WriteImage(label, './temp/' + str(i) + '.nii.gz')
		# print(k.shape)
		k = k.numpy()
		k = np.squeeze(k)
		# print(k.shape)
		if x.shape[0] !=8:
			img2[id * batchsize:, :, :] = k
		else:
			img2[id * batchsize:(id + 1) * batchsize, :, :] = k
		id = id + 1
		# img = sitk.GetImageFromArray(k)
		# sitk.WriteImage(img, './temp/img' + str(i) + '.nii.gz')
		if i < num: i += 1
	predict = sitk.GetImageFromArray(predict)
	predict.SetDirection(origin_image.GetDirection())
	predict.SetOrigin([0, 0, 0])
	predict.SetSpacing([1, 1, 1])
	predict=sitk.Cast(predict,sitk.sitkUInt8)
	sitk.WriteImage(predict, './result/predict/'+path)
	# predict=img2
	# predict = sitk.GetImageFromArray(predict)
	# predict.SetDirection(origin_image.GetDirection())
	# sitk.WriteImage(predict, './temp/img.nii.gz')
	shutil.copy(origin_path + path, './result/image/'+path)
	shutil.copy(origin_seg_path + path2, './result/seg/'+path)
	aver_dice = np.mean(dices)
	print('aver_dice=%f' % (aver_dice))


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
	logging = getLog(args)
	print('**************************')
	print('models:%s\nepoch:%s\nbatch size:%s' % \
	      (args.arch, args.epoch, args.batch_size))
	print('**************************')
	model = getModel(args)
	weight=r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	# weight=r'./saved_model/latest' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	model.load_state_dict(torch.load(weight, map_location='cuda'))
	train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
	# loss function BCE  二进制交叉熵   Sigmoid +BCELoss.
	from dice_loss import SoftDiceLoss
	
	criterion0 = torch.nn.BCELoss()
	criterion = SoftDiceLoss()
	
	# 优化器 Adam  SGD  lr_scheduler学习率调整
	optimizer = optim.Adam(model.parameters(), lr=0.0006)
	# scheduler =None
	# optimizer = optim.SGD(model.parameters(),lr=0.001)
	# optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.99,verbose=True)
	for path in os.listdir('E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/test_one/image/'):
		Volume_test(path)
	exit(0)
	# Volume_test('06.nii.gz')
	# args.action = 'test'
	args.action = 'train'
	if 'train' in args.action:
		train(model, criterion0, criterion, optimizer, train_dataloaders, val_dataloaders, args, scheduler=scheduler)
		torch.save(model.state_dict(), r'./saved_model/final' + str(args.arch) + '_' + str(args.batch_size) +
		           '_' + str(args.epoch) + '.pth')
	if 'test' in args.action:
		K_test(test_dataloaders, save_predict=True)
