import os
import torch

class Processor_Hparams():

    def __init__(self):
        ################ preprocess  ###################################
        self.seglen = 128 ## 语音片段的长度/帧数
        self.scaler = True ## 是否归一化特征集合
        self.trim_db = 20  ##  静音消除参数
        self.coef = 0.97 ## 预加重系数
        self.n_fft = 1024  ## 提取出 513维度的傅里叶谱，再转为80维度 melspec
        self.win_length = 1024 ## 帧长
        self.hop_length = 220  ## 帧移
        self.sample_rate = 22050
        self.f_min = 0
        self.f_max = 11025 ## 谱（还未取对数的时候）中的数值 通常最大值设为采样率的一半
        self.n_mels = 40 ## mel频谱的维度
        self.n_mfcc = 40 ## mfcc的维度
        self.train_ratio = 0.9 ## 训练集比例

class Training_Hparams():

    def __init__(self,feature_set='feature_MFCC', n_features=None, model:str = 'CNN_BiLSTM'):
        '''
        :param feature_set: 特征集的名称
        :param n_features: 特征集的维度
        :param model: 模型名称
        '''
        ################ feature set   ###################################
        if n_features is None:
            raise ValueError("n_features is None")
        else:
            self.n_features = n_features
        if model in ('CNN_BiLSTM', 'CNN', 'CNN_LSTM'):
                self.model = model
        else:
            raise ValueError("model is not in set('CNN_BiLSTM', 'CNN', 'CNN_LSTM')")
        ################ Trainer  ###################################
        self.hidden_size = 128
        self.num_layers = 2
        self.total_iters = 2000 ## 总共训练步骤
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_every = 200  ## 每隔200次 保存一次模型
        self.eval_every = 100  ## 每隔50次，进行一次验证
        ################ dataset / loader  ###################################
        self.train_ratio  = 0.9 ## 切分训练集、测试集的比例
        self.frame_length = 128  ###  训练时音频的特征帧数
        self.batchsize  = 32 ## 训练的batchsize
        #################  Adam optimizer  ##############################
        self.lr_start = 0.001 # 初始学习率
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.amsgrad = True
        self.gamma = 0.95  ## 学习率衰减系数后续可能要调整
        self.lr_dacay_every = 100  ## 每隔多少次更新一次学习率
        ####################### model params #########################################
        self.num_classes = 2  ## 数据集的分类数量。
        ################################################################
        ## root dir
        self.root_dir = './' + str(model)
        ## dirs create
        self.version = feature_set
        self.version_dir = self.root_dir + "/" + self.version
        self.model_savedir = self.version_dir + "/" + 'checkpoints_{}'.format(feature_set)
        if not os.path.exists(self.version):
            os.makedirs(self.version)
        if not os.path.exists(self.version_dir):
            os.makedirs(self.version_dir)
        if not os.path.exists(self.model_savedir):
            os.makedirs(self.model_savedir)
        ## files create
        self.hp_filepath = self.version_dir + "/" + 'hparams_{}.txt'.format(feature_set)
        self.log_train_filepath = self.version_dir + "/" + 'logs_train_{}.txt'.format(feature_set)
        self.log_eval_filepath = self.version_dir + "/" + 'logs_eval_{}.txt'.format(feature_set)
        if not os.path.exists(self.hp_filepath):
            open(self.hp_filepath, 'w').close()
        if not os.path.exists(self.log_train_filepath):
            open(self.log_train_filepath, 'w').close()
        if not os.path.exists(self.log_eval_filepath):
            open(self.log_eval_filepath, 'w').close()
        print(f"*"*20 + str({feature_set}) + "模型和日志存储文件生成完毕" + "*"*20)

if __name__ == "__main__":
    pass
