import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet,resnet34_unet
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

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--epoch", type=int, default=203)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='cenet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument('--dataset', default='E:/lijin/pancreas_tumor_seg/pancreas_tumor_2d/', help='path')
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    #args = parse.parse_args()
    args = parse.parse_args(args=[])
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1,pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args,3,1).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(3,1).to(device)
    if args.arch == 'segnet':
        model = SegNet(3,1).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3,1).to(device)
    if args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    train_dataset =Dataset(r"train",path=args.dataset, transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = Dataset(r"val",path=args.dataset, transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataset = Dataset(r"test",path=args.dataset, transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_dice,val_dataloaders):
    print('**************************')
    print('Val')
    print('**************************')
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        dices=[]
        num = len(val_dataloaders)  #验证集图片的总数
        for x, l,pic,mask in tqdm(val_dataloaders):
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            l=l.numpy()
            l=np.squeeze(l)
            img_y[img_y>0.5]=1
            img_y[img_y <= 0.5] = 0
            # print(np.min(l),np.max(l))
            # print(np.min(img_y),np.max(img_y))
            y_true_f = l.flatten()
            y_pred_f = img_y.flatten()
            intersection = sum(y_true_f * y_pred_f)
            smooth=1
            dice=(2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
            # print(i,dice)
            dices.append(dice)
            if i < num:i+=1
        aver_dice=np.mean(dices)
        print('aver_dice=%f' % (aver_dice))
        print('best_dice=%f' % (best_dice))
        if aver_dice > best_dice:
            print('aver_dice:{} > best_dice:{}'.format(aver_dice,best_dice))
            best_dice = aver_dice
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+'best_'+str(args.epoch)+'.pth')
        torch.save(model.state_dict(), r'./saved_model/latest'  '_' + str(args.epoch) + '.pth')
        return best_dice,aver_dice
from tqdm import tqdm
def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_dice=0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    dice_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        # print(dt_size)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in tqdm(train_dataloader):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output, labels)
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_dice,aver_dice = val(model,best_dice,val_dataloader)
        # iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        # hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'dice', dice_list)
    # metrics_plot(args,'hd',hd_list)
    return model

def k_test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0],predict)
            dice_total += dice

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            #print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            #print(mask_path[0])
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    plt.savefig(dir +'/'+ mask_path[0].split('/')[-1])
            #plt.pause(0.01)
            print('iou={},dice={}'.format(iou,dice))
            if i < num:i+=1   #处理验证集下一张图
        #plt.show()
        print('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        #print('M_dice=%f' % (dice_total / num))

if __name__ =="__main__":
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
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        k_test(test_dataloaders, save_predict=True)