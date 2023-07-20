import os,shutil
src=r'./saved_model/resunet_cbam2_16_203.pth'
# dst='./saved_model/Ablation/resunet_M_4.pth'
dst='./saved_model/stage1/resunet_PSAM_4_n.pth'
try:
    # os.rename(src, dst)
    shutil.copy(src, dst)
except Exception as e:
    print(e)
    print('rename file fail\r\n')
else:
    print('rename file success\r\n')