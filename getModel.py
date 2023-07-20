import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	if args.arch == 'attention_unet':
		from attention_unet import AttU_Net
		model = AttU_Net().to(device)
	if args.arch == 'resunet_cbam2':
		from UNet import resunet_cbam2
		model = resunet_cbam2().to(device)
	if args.arch == 'cenet_cbam':
		from cenet import CE_Net_CBAM
		model = CE_Net_CBAM().to(device)
	if args.arch == 'epsaunet2':
		from Epsaunet import Epsaunet
		model = Epsaunet().to(device)
	# from torchsummary import summary
	# summary(model, (3, 512, 512))
	# exit(0)
	return model