import torch
import numpy as np
from torch.utils.data import  Dataset
from hparms import Training_Hparams

class MydataSet(Dataset):

    def __init__(self,version_dir:str,txt_name:str):
        txt_path = version_dir + "/" + txt_name
        ## print(version_dir)
        self.scripts = []
        with open(txt_path,encoding='utf-8') as f :
            for l in f.readlines():
                self.scripts.append(l.strip('\n'))
        self.L = len((self.scripts))

        ## 建立语音标签的查找表每条语音的真实标签类别就根据这个表去查找
        ## 对于分类数据集，给定音频类别集合set['pathology', 'healthy']
        ## 为了crossentroy 损失函数的要求，则需要下标映射查找表
        self.audio_class_names = list(set(line.replace('\\', '/').split("/")[-2] for line in self.scripts))
        print("audio_class:{}".format(self.audio_class_names))
        ## line.replace('\\', '/').split("/")[-2]为路径的第二个元素
        ## 例如路径：feature_set_1\\healthy\\healthy-female-153-i_l.npy
        ## 这里先取set再取list是为了去重得到lits['pathology', 'healthy']

        pass

    def __getitem__(self, index):
        src_path = self.scripts[index]
        src_npy = np.load(str(src_path)) # 读取时候把npy文件路径变成字符串
        audio_class = str(src_path).replace('\\', '/').split("/")[-2] # 从路径的子文件夹中提取出类别
        label  = torch.tensor(self.audio_class_names.index(audio_class)) # 根据查找表得到类别的下标[0,1]
        return torch.FloatTensor(src_npy),label # 返回tensor类型的npy和label

    def __len__(self):
        return self.L # 返回数据集的长度

if __name__=="__main__":

    hp = Training_Hparams(feature_set="feature_set_4")
    print(hp.version_dir)
    meldataSet = MydataSet(version_dir=hp.version_dir,txt_name="train.txt")
    print(meldataSet[2]) ## getitem方法返回元组，第一个元素是npy，第二个元素是label
    print(f"npy维度",meldataSet[2][0].shape)
    print(f"label标签",meldataSet[2][1])
    pass
