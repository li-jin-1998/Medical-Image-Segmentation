import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from metrics import *
from torchvision.transforms import transforms
import shutil

def getArgs():
	parse = argparse.ArgumentParser()
	parse.add_argument('--deepsupervision', default=0)
	# parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
	parse.add_argument("--epoch", type=int, default=203)
	parse.add_argument('--arch', '-a', metavar='ARCH', default='resunet_cbam2',
	                   help='unet/cenet/resunet/r2unet/m2unet/epsaunet/resunet_cbam2/attention_unet')
	parse.add_argument("--batch_size", type=int, default=16)
	parse.add_argument('--dataset', default='E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/', help='path')
	# parse.add_argument("--ckp", type=str, help="the path of model weight file")
	# parse.add_argument("--log_dir", default='result/log', help="log dir")
	# parse.add_argument("--threshold", type=float, default=None)
	# args = parse.parse_args()
	args = parse.parse_args(args=[])
	return args

from getModel import getModel
def window_transform(ct_array, windowWidth, windowCenter, normal=True):
	minWindow = float(windowCenter) - 0.5 * float(windowWidth)
	newimg = (ct_array - minWindow) / float(windowWidth)
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	if not normal:
		newimg = (newimg * 255).astype('float32')
	return newimg


def Volume_test(model,path):
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
	print(batchsize)
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
	print('**************************')
	print('models:%s\nepoch:%s\nbatch size:%s' % \
	      (args.arch, args.epoch, args.batch_size))
	print('**************************')
	model = getModel(args)
	# weight = r'E:\lijin\pancreas_tumor_seg\pancreas_segmentation\saved_model\Ablation/resunet_P_1.pth'
	# weight=r'E:\lijin\pancreas_tumor_seg\pancreas_segmentation\saved_model\stage1\resunet_PSAM_4.pth'
	weight = r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	# weight=r'./saved_model/latest' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.epoch) + '.pth'
	model.load_state_dict(torch.load(weight, map_location='cuda'))
	result_path='./result/'
	if os.path.exists(result_path):
		shutil.rmtree(result_path)
	if not os.path.exists(result_path):
		os.mkdir(result_path)
		os.mkdir(result_path+'predict')
		os.mkdir(result_path + 'image')
		os.mkdir(result_path+'seg')

	for path in os.listdir('E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/test_one/image/'):
		Volume_test(model,path)
	from Compute_metrics import dice, volumeMetrics
	import os
	import SimpleITK as sitk
	origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/seg/'
	label_paths = os.listdir(origin_path)
	dice1 = []
	for path in label_paths:
		print(path)
		label = sitk.ReadImage(origin_path + path, sitk.sitkUInt8)
		predict = sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/predict/' + path,
		                         sitk.sitkUInt8)
		dice1.append(volumeMetrics(label, predict))
		print('Pre', volumeMetrics(label, predict))
	print(np.mean(dice1), np.std(dice1))
