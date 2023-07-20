import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CrossEntropyLabelSmooth2(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth2, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.to('cuda:0')
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
import torch
import torch.nn as nn
from IPython import embed


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    
    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num
    
    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            
            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)
        
        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        
        return loss.mean()


if __name__ == '__main__':
    batch_size = 4  # 首先是对应的批尺寸
    num_class = 8 # 对应的分类数是8
    loss_class = CELoss(label_smooth=0.1,class_num=num_class)
    input = torch.randn([batch_size,num_class])  # 这里是来随机生成一批数据的
    label = torch.randint(0, num_class-1, [batch_size]) # 用于生成指定范围的整数，也就是对对应的标签 tensor([4, 5, 3, 3])
    print(label)
    print(loss_class(input, label))
