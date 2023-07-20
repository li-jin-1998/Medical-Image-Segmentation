import shutil,os,glob
origin_result_path = './result\\'
# result_path = './Ablation_result/resunet_M_4/'


# origin_result_path = './final_result2/cenet_0\\'
result_path = './final_result2/resunet_PSAM_4/'
if not os.path.exists(result_path):
	os.mkdir(result_path)
	os.mkdir(result_path + 'predict')
	os.mkdir(result_path + 'image')
	os.mkdir(result_path + 'seg')
paths=glob.glob(origin_result_path+'*/*')
print(len(paths))
for path in paths:
	path=str(path)
	print(path,path.replace(origin_result_path,result_path))
	shutil.copy(path,path.replace(origin_result_path,result_path))


