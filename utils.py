import os
import torch
import opensmile
import audiofile
import numpy as np
import random
import scipy.io as sio
import pickle
from matplotlib import pyplot as plt


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for reproducibility
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # CuDNN brings nondeterministic results with same seed for RNN & LSTM

def pad(x, seglen, mode='constant'):
    '''
    :param x: ndarry [D,T]
    :param seglen: padding长度
    :param mode: padding模式
    '''
    pad_len = seglen - x.shape[1]
    y = np.pad(x, pad_width=((0, 0), (0, pad_len)), mode=mode)
    return y

def segment(x, seglen=100):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    ## 该函数将feature[40,len] padding到固定长度 [40,seglen]
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    ## 过长的feature进行截断
    else:
        r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:,r:r+seglen]
    return y

def scaler(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return m

def extract_phonation(the_wavpath:str,seglen:int):
    the_wavpath = str(the_wavpath)
    the_matpath = str(the_wavpath).replace('wav','mat')
    ## 读取mat文件储存的频谱斜度参数
    data=sio.loadmat(the_matpath)
    H1H2C = data["H1H2c"].T
    H2H4C = data["H2H4c"].T
    H1A1C = data["H1A1c"].T
    H1A2C = data["H1A2c"].T
    H1A3C = data["H1A3c"].T
    spectral_tilts = np.concatenate((H1H2C, H2H4C, H1A1C, H1A2C,H1A3C), axis=0)
    spectral_tilts[np.isnan(spectral_tilts)] = 0
    spectral_tilts = segment(spectral_tilts, seglen=seglen)
    ## 读取CPP参数和嗓音音质参数
    CPP = data["CPP"].T #显示属性值#
    CPP[np.isnan(CPP)] = 0
    CPP = segment(CPP, seglen=seglen)
    SOE = data["soe"].T #显示属性值#
    SOE[np.isnan(SOE)] = 0
    SOE = segment(SOE, seglen=seglen)
    signal, sampling_rate = audiofile.read(the_wavpath)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    feature = smile.process_signal(signal,sampling_rate)
    jitter = feature['jitterLocal_sma3nz']
    jitter = jitter.to_numpy().reshape(1,-1)
    shimmer = feature['shimmerLocaldB_sma3nz']
    shimmer = shimmer.to_numpy().reshape(1,-1)
    HNR = feature['HNRdBACF_sma3nz']
    HNR = HNR.to_numpy().reshape(1,-1)
    voicing_parms = np.concatenate((jitter,shimmer,HNR),axis=0)
    voicing_parms = segment(voicing_parms, seglen=seglen)
    voicing_parms = np.concatenate((voicing_parms,CPP,SOE),axis=0)
    ## 合并mel频谱和频谱斜度和嗓音音质参数
    phonation_parms = np.concatenate((spectral_tilts,voicing_parms), axis=0)
    return phonation_parms

def plot_results(model_name:str ="CNN_BiLSTM",feature_name:str = "feature_Fbank_and_Phonation",query="loss"):
    '''
    :param model_name: 模型名称 
    :param feature_name: 版本号/特征集合名称
    :param query: 想要查询的关键字
    "lr":self.current_lr
    "loss":batch_loss.item()
    "acc":batch_accuracy
    '''
    pickle_saving_path = model_name + "/" + feature_name + "/" + 'running_loss_{}.pickle'.format(feature_name)
    if os.path.exists(pickle_saving_path) == False:
        print("pickle文件不存在")
    with open(pickle_saving_path ,'rb') as f :
        log = pickle.load(f)
    ################
    for k,v in log.items():
        if query in k:
            results = log[k]
            plt.figure()
            plt.title(k)
            plt.xlabel(os.path.basename(pickle_saving_path))
            plt.plot(results)
            plt.show()

if __name__ == '__main__':

    plot_results("CNN_BiLSTM", "feature_MFCC_and_Phonation", query="acc")