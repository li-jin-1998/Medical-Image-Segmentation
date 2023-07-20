from Compute_metrics import dice,volumeMetrics
import os
import numpy as np
import SimpleITK as sitk
origin_path = 'E:/lijin/pancreas_tumor_seg/pancreas_segmentation/Ablation_result/'
nets=os.listdir(origin_path)
Dice={}
paths=[]
for net in nets:
    print(net)
    net_path=origin_path+net
    paths=os.listdir(net_path+'/seg/')
    Dice[str(net)]=[]
    for path in paths:
        label=sitk.ReadImage(net_path+'/seg/'+path,sitk.sitkUInt8)
        predict=sitk.ReadImage(net_path+'/predict/'+path,sitk.sitkUInt8)
        Dice[str(net)].append(volumeMetrics(label,predict))
        # print(path,volumeMetrics(label,predict))
    # print(np.mean(Dice[str(net)]) * 100, np.std(Dice[str(net)]) * 100)
    # print(len(Dice[str(net)]))
    print('{:.2f}±{:.2f}'.format(np.mean(Dice[str(net)])*100, np.std(Dice[str(net)])*100))
    # print('{:.2f}±{:.2f}'.format(round(np.mean(Dice[str(net)])*100,2), round(np.std(Dice[str(net)])*100,2)))
# with open('./result.txt', "w+") as f:
#     f.write('path'+'\t'*2)
#     for net in nets:
#         if len(net) > 8:
#             f.write(net + '\t' )
#         else:
#             f.write( net + '\t'*2)
#     f.write('\n')
#     for i,path in enumerate(paths):
#         f.write(path+'\t'*2)
#         for net in nets:
#             # print(str(Dice[str(net)][i]))
#             f.write('{:.4f}'.format(Dice[str(net)][i]) + '\t'*2)
#         f.write('\n')
# f.close()
