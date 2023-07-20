import SimpleITK as sitk
import os
import numpy as np
dir='./result/result_label/'  #预测文件夹
dir3='D:/hy/380-80/liver/test/label/'#实际label
paths=os.listdir(dir)
num=len(paths)
paths3=os.listdir(dir3)[:num]
n=len(paths)
a=np.zeros((n,512,512))
print(a.shape)
i=0
for path,path3 in zip(paths,paths3):
    image=sitk.ReadImage(dir+path,sitk.sitkUInt8)
    image3 = sitk.ReadImage(dir3 + path3, sitk.sitkUInt8)
    array=sitk.GetArrayFromImage(image)
    array[array>127]=1
    array3 = sitk.GetArrayFromImage(image3)
    array3[array3 > 127] = 2
    a[i,:,:]=array+array3
    i=i+1
label=sitk.GetImageFromArray(a)
sitk.WriteImage(label,'./label.nii.gz')

dir2='D:/hy/380-80/liver/test/ct/'  #实际image
paths=os.listdir(dir2)[:num]
n=len(paths)
b=np.zeros((n,512,512))
print(a.shape)
for i,path in enumerate(paths):
    image=sitk.ReadImage(dir2+path,sitk.sitkFloat32)
    array=sitk.GetArrayFromImage(image)
    b[i,:,:]=array
image=sitk.GetImageFromArray(b)
sitk.WriteImage(image,'./image.nii.gz')