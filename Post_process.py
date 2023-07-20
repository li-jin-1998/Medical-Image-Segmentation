import numpy as np
from scipy import ndimage
import SimpleITK as sitk

def filter_largest_volume(seg):
    '''只保留分割标签中最大的连通块
    seg: numpy数组
    '''
    # TODO 需要能处理vol 是一个batch的情况
    seg, num = ndimage.label(seg, np.ones([3,3,3]))
    maxi = 0
    maxnum = 0
    for i in range(1,num):
        count = seg[seg == i].size
        if count > maxnum:
            maxi = i
            maxnum = count
    seg[seg != maxi ] = 0
    seg[seg == maxi] = 1
    return seg

def post(fixed,segmentation):
    label = sitk.BinaryFillhole(segmentation)
    # label = sitk.BinaryDilate(label, (1, 1, 1))
    label = sitk.BinaryErode(label, (1, 1, 1))
    label = sitk.BinaryDilate(label, (1, 1, 1))
    labels = sitk.BinaryDilate(label, (1, 1, 1))
    # print(fixed.GetOrigin(),labels.GetOrigin())
    new = sitk.Mask(fixed, labels, outsideValue=0, maskingValue=0)
    # sitk.WriteImage(new,'./result/new.nii.gz')
    LabelStatistics = sitk.LabelStatisticsImageFilter()
    LabelStatistics.Execute(fixed, sitk.Cast(label, sitk.sitkUInt8))
    median = LabelStatistics.GetMedian(1)
    # print(median)
    image = sitk.BinaryThreshold(new, lowerThreshold=median - 0.13, upperThreshold=median + 0.09, insideValue=1,
                                 outsideValue=0)
    image = sitk.BinaryFillhole(image)
    vectorRadius = (1, 1, 1)
    kernel = sitk.sitkCross
    image = sitk.BinaryMorphologicalClosing(image, vectorRadius, kernel)
    image = sitk.Cast(image, sitk.sitkUInt8)
    label = sitk.And(label, image)
    label = sitk.BinaryFillhole(label)
    label = sitk.BinaryMorphologicalClosing(label, vectorRadius, kernel)
    label = sitk.BinaryErode(label, (1, 1, 1))
    label = sitk.BinaryDilate(label, (1, 1, 1))
    return label
def simple_post(segmentation):
    label = sitk.BinaryFillhole(segmentation)
    # vectorRadius = (2,2,2)
    # kernel = sitk.sitkCross
    # label = sitk.BinaryMorphologicalClosing(label, vectorRadius, kernel)
    # label = sitk.BinaryDilate(label, (3, 3, 3))
    kernel=(2,2,2)
    label = sitk.BinaryDilate(label, kernel)
    label = sitk.BinaryErode(label, kernel)
    # label = sitk.BinaryDilate(label, kernel)
    return label
from Compute_metrics import dice,volumeMetrics
import os
origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/seg/'
label_paths=os.listdir(origin_path)
dice1=[]
dice2=[]
dice3=[]
dice4=[]
for path in label_paths:
    print(path)
    label=sitk.ReadImage(origin_path+path,sitk.sitkUInt8)
    predict=sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/predict/'+path,sitk.sitkUInt8)
    # fixed=sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/image/'+path,sitk.sitkFloat32)
    # fixed.SetDirection(label.GetDirection())
    # fixed.SetOrigin([0, 0, 0])
    # fixed.SetSpacing([1, 1, 1])
    dice1.append(volumeMetrics(label,predict))
    print('Pre',volumeMetrics(label,predict))
    # array=sitk.GetArrayFromImage(predict)
    # seg=filter_largest_volume(array)
    # seg=sitk.GetImageFromArray(seg)
    # seg.SetDirection(label.GetDirection())
    # seg=sitk.Cast(seg,sitk.sitkUInt8)
    # dice2.append(volumeMetrics(seg,label))
    # print('Post1:',volumeMetrics(seg,label))
    # seg1=simple_post(predict)
    # seg1=post(fixed,predict)
    # dice3.append(volumeMetrics(seg1,label))
    # print('Post2:',volumeMetrics(seg1,label))
    # seg2=simple_post(seg)
    # # seg2 = post(fixed, seg)
    # dice4.append(volumeMetrics(seg2, label))
    # print('Post:',volumeMetrics(seg2,label))
print(np.mean(dice1),np.std(dice1))
# print(np.mean(dice2),np.std(dice2))
# print(np.mean(dice3),np.std(dice3))
# print(np.mean(dice4),np.std(dice4))
# label=sitk.ReadImage('./temp/seg.nii.gz',sitk.sitkUInt8)
# fixed=sitk.ReadImage('./temp/image.nii.gz',sitk.sitkFloat32)
# predict=sitk.ReadImage('./temp/predict.nii.gz',sitk.sitkUInt8)
# # print(predict.GetSpacing(),label.GetSpacing())
# print('Pre:',volumeMetrics(predict,label))
# array=sitk.GetArrayFromImage(predict)
# seg=filter_largest_volume(array)
# seg=sitk.GetImageFromArray(seg)
# seg.SetDirection(label.GetDirection())
# seg=sitk.Cast(seg,sitk.sitkUInt8)
# print('Post1:',volumeMetrics(seg,label))
# sitk.WriteImage(seg,'./temp/predict2.nii.gz')
# seg1=simple_post(predict)
# # seg1=post(fixed,predict)
# print('Post2:',volumeMetrics(seg1,label))
# sitk.WriteImage(seg1,'./temp/predict3.nii.gz')
# seg2=simple_post(seg)
# print('Post:',volumeMetrics(seg2,label))
# sitk.WriteImage(seg2,'./temp/predict4.nii.gz')