from Compute_metrics import dice,volumeMetrics
import os
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import pandas as pd
origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/seg/'
label_paths=os.listdir(origin_path)
dice1=[]
Name=[]
for path in label_paths:
    print(path)
    Name.append(path)
    label=sitk.ReadImage(origin_path+path,sitk.sitkUInt8)
    predict=sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/predict/'+path,sitk.sitkUInt8)
    fixed=sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/image/'+path,sitk.sitkFloat32)
    fixed.SetDirection(label.GetDirection())
    fixed.SetOrigin([0, 0, 0])
    fixed.SetSpacing([1, 1, 1])
    dice1.append(volumeMetrics(label,predict))
    print('Pre',volumeMetrics(label,predict))
print(np.mean(dice1),np.std(dice1))
