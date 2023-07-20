import os
import time

import SimpleITK as sitk
import numpy as np
def dice(image1,image2,average=True):
    scores=[]
    image1.SetOrigin([0, 0, 0])
    image2.SetOrigin([0, 0, 0])
    lom_filter = sitk.LabelOverlapMeasuresImageFilter()
    lom_filter.Execute(image1, image2)
    scores.append(lom_filter.GetDiceCoefficient())
    return scores if not average else np.mean(scores)

def dice2(predict,label):
    predict=sitk.GetArrayFromImage(predict)
    label = sitk.GetArrayFromImage(label)
    y_true_f = predict.flatten()
    y_pred_f = label.flatten()
    intersection = sum(y_true_f * y_pred_f)
    smooth = 1
    dice = (2. * intersection + smooth) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f) + smooth)
    return dice
def volumeMetrics(imFixed, imMoving):
    """
    DSC, VolOverlap, FracOverlap, TruePosFrac, TrueNegFrac, FalsePosFrac, FalseNegFrac
    """
    arrFixed = sitk.GetArrayFromImage(imFixed).astype(bool)
    arrMoving = sitk.GetArrayFromImage(imMoving).astype(bool)

    arrInter = arrFixed & arrMoving
    arrUnion = arrFixed | arrMoving

    voxVol = np.product(imFixed.GetSpacing())/1000. # Conversion to cm^3

    # 2|A & B|/(|A|+|B|)
    DSC =  (2.0*arrInter.sum())/(arrFixed.sum()+arrMoving.sum())

    #  |A & B|/|A | B|
    # FracOverlap = arrInter.sum()/arrUnion.sum().astype(float)
    # VolOverlap = arrInter.sum() * voxVol
    #
    # TruePos = arrInter.sum()
    # TrueNeg = (np.invert(arrFixed) & np.invert(arrMoving)).sum()
    # FalsePos = arrMoving.sum()-TruePos
    # FalseNeg = arrFixed.sum()-TruePos
    #
    # #
    # TruePosFrac = (1.0*TruePos)/(TruePos+FalseNeg)
    # TrueNegFrac = (1.0*TrueNeg)/(TrueNeg+FalsePos)
    # FalsePosFrac = (1.0*FalsePos)/(TrueNeg+FalsePos)
    # FalseNegFrac = (1.0*FalseNeg)/(TruePos+FalseNeg)
    #
    #
    # return DSC, VolOverlap, FracOverlap, TruePosFrac, TrueNegFrac, FalsePosFrac, FalseNegFrac
    return DSC

# origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/seg/'
# label_paths=os.listdir(origin_path)
# for path in label_paths:
#     label=sitk.ReadImage(origin_path+path)
#     predict=sitk.ReadImage('E:/lijin/pancreas_tumor_seg/pancreas_segmentation/result/predict/'+path)
#     print(path,volumeMetrics(label,predict))

# predict=sitk.ReadImage('./temp/predict2.nii.gz',sitk.sitkUInt8)
# label=sitk.ReadImage('./temp/seg.nii.gz',sitk.sitkUInt8)
# t=time.time()
# print(dice(predict,label))
# print(time.time()-t,'s')