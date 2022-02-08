import os 
from torchvision import datasets, models, transforms
import PIL
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import skimage
from PIL import Image
WORKING_PATH = 'D:/1A文件资料/浙大项目/copycat/tensorflow_shufflenet_aid.txt'
image_path ='D:/1A文件资料/浙大项目/DataSet/AID/train/'


def dic(WORKING_PATH,method):
	# pre_label=''
	data_set=dict()
	with open(WORKING_PATH,'r')as file:
		for line in file:
			cont=eval(line)
			image_name=cont['image_name']
			credit=cont['credit']
			cls_number=len(list(cont['credit'].keys()))
			if method=='copycat_label':
				n=np.random.randint(1,101)
				s=np.random.randint(0,cls_number)
				if n<=5:
					pre_label=s
				else:
					pre_label=cont['pre_label']
			else:
				pre_label = cont['pre_label']
			if os.path.isfile(os.path.join(image_path,image_name)):
				if image_name in data_set:
					pass
				data_set[image_name]={"credit":credit,'pre_label':pre_label}
	# cls_number=len(list(cont['credit'].keys()))
	return data_set,cls_number

class my_data_set(Dataset):
	def __init__(self, data,method):
		self.data=data
		self.image_ids=list(self.data.keys())
		self.method=method
		for id in data.keys():
			self.data[id]["image_path"] = os.path.join(image_path,str(id))
	def __image_loader(self,id):
		path= self.data[id]["image_path"]
		img_pil =  PIL.Image.open(path)
		if self.method=='copycat_gs' or 'Copycat_Gs'or'CopyCat_Gs':
			n=np.random.randint(1,101)
			if n<=5:
				img_arr = np.array(img_pil)
				img_arr = skimage.util.random_noise(img_arr, mode='gaussian', var=0.01)
				img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
			else:
				pass
		else:
			pass
		transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
		img_tensor = transform(img_pil)
		return img_tensor
	# def text_loader(self,data):
	# 	return self.data["pre_label"]
	def __group_loader(self,id):
		ind=list(self.data[id]['credit'].keys())
		group=ind.index(self.data[id]['pre_label'])
		# if gs=='True':
		# 	n=np.random.randint(1,101)
		# 	if n<=5:
		return group
	def __getitem__(self,index):
		id=self.image_ids[index]
		img = self.__image_loader(id)
		# text_index = self.__text_index_loader(id)
		# image_feature = self.__image_feature_loader(id)
		group =self.__group_loader(id)
		# credit=self.data[id]['credit']
		credit = np.array(list(self.data[id]['credit'].values()))
		return img,group,credit
	def __len__(self):
		return len(self.image_ids)
#
