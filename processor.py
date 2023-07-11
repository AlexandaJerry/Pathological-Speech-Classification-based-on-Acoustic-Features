import os
import random
import librosa
import warnings
import numpy as np
from hparms import Processor_Hparams
from utils import same_seeds,segment,scaler,extract_phonation
warnings.filterwarnings('ignore')

def extract_feature_Melspec(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维Melspectrogram
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)        
        melspec = librosa.feature.melspectrogram(wavform,sr=hp.sample_rate,hop_length=hp.hop_length,
        n_fft=hp.n_fft,win_length=hp.win_length,n_mels=hp.n_mels) ## 按照超参数进行mel提取
        melspec = segment(melspec, seglen=hp.seglen)
        melspec = scaler(melspec,Scale=hp.scaler)
        np.save(the_npypath, melspec)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],melspec.shape[-1],melspec.shape[0]))
    print("*"*20+"finished extracting feature Melspec"+"*"*20)

def extract_feature_Melspec_and_Phonation(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维Melspectrogram和10维发声参数
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)        
        melspec = librosa.feature.melspectrogram(wavform,sr=hp.sample_rate,hop_length=hp.hop_length,
        n_fft=hp.n_fft,win_length=hp.win_length,n_mels=hp.n_mels) ## 按照超参数进行mel提取
        melspec = scaler(segment(melspec, seglen=hp.seglen))
        phonation_parms = scaler(extract_phonation(the_wavpath,seglen=hp.seglen))
        feature_set = np.concatenate((melspec,phonation_parms),axis=0)
        np.save(the_npypath, feature_set)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],feature_set.shape[-1],feature_set.shape[0]))
    print("*"*20+"finished extracting feature Melspec and Phonation Parms"+"*"*20)

def extract_feature_MFCC(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维MFCC
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)
        mfcc = librosa.feature.mfcc(wavform, sr=hp.sample_rate, hop_length=hp.hop_length,
        n_fft=hp.n_fft,win_length=hp.win_length, n_mfcc=hp.n_mfcc)
        mfcc = segment(mfcc, seglen=hp.seglen)
        mfcc = scaler(mfcc,Scale=hp.scaler)
        np.save(the_npypath, mfcc)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],mfcc.shape[-1],mfcc.shape[0]))
    print("*"*20+"finished extracting feature MFCC"+"*"*20)

def extract_feature_MFCC_and_Phonation(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维MFCC和10维发声参数
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)
        mfcc = librosa.feature.mfcc(wavform, sr=hp.sample_rate, hop_length=hp.hop_length,
        n_fft=hp.n_fft,win_length=hp.win_length, n_mfcc=hp.n_mfcc)
        mfcc = scaler(segment(mfcc, seglen=hp.seglen))
        phonation_parms = scaler(extract_phonation(the_wavpath,seglen=hp.seglen))
        feature_set = np.concatenate((mfcc,phonation_parms),axis=0)
        np.save(the_npypath, feature_set)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],feature_set.shape[-1],feature_set.shape[0]))
    print("*"*20+"finished extracting feature MFCC and Phonation Parms"+"*"*20)

def extract_feature_Fbank(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维Fbank
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)
        x_stft = librosa.stft(wavform,n_fft=hp.n_fft,hop_length=hp.hop_length,
        win_length=hp.win_length)
        spc = np.abs(x_stft).T
        mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels = hp.n_mels,
        fmin = hp.f_min, fmax=hp.f_max)
        fbank = np.log10(np.maximum(1e-8, np.dot(spc, mel_basis.T)))
        fbank = fbank.T
        fbank = segment(fbank, seglen=hp.seglen)
        fbank = scaler(fbank,Scale=hp.scaler)
        np.save(the_npypath, fbank)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],fbank.shape[-1],fbank.shape[0]))
    print("*"*20+"finished extracting feature fbank"+"*"*20)

def extract_feature_Fbank_and_Phonation(hp:Processor_Hparams,wav_dir:str,feature_dir:str):
    '''
    :param hp: Processor_Hparams
    :提取40维Fbank和10维发声参数
    '''
    wav_file=[]
    wav_dir = wav_dir
    feature_dir = feature_dir
    ## 制作特征提取文件夹
    for dirpath, dirnames, filenames in os.walk(wav_dir): 
        dn = dirpath.replace(wav_dir, feature_dir)
        if not os.path.exists(dn):
            os.makedirs(dn)
    ## 搜索音频文件夹下所有音频路径写入wav_file
    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.wav':  
                wav_file.append(os.path.join(dirpath, file)) 
    total = len(wav_file) ## 总语音数            
    count = 0 ## 任务进度
    for wav in wav_file:
        count += 1
        the_wavpath = str(wav)
        ## 将特征存储为numpy文件
        the_npypath = str(wav).replace(wav_dir,feature_dir).replace('wav','npy')
        wavform,_ = librosa.load(the_wavpath)
        wavform,_ = librosa.effects.trim(wavform,top_db=hp.trim_db)  ## 静音消除
        wavform = librosa.effects.preemphasis(wavform, coef=hp.coef, zi=None)
        x_stft = librosa.stft(wavform,n_fft=hp.n_fft,hop_length=hp.hop_length,
        win_length=hp.win_length)
        spc = np.abs(x_stft).T
        mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels = hp.n_mels,
        fmin = hp.f_min, fmax=hp.f_max)
        fbank = np.log10(np.maximum(1e-8, np.dot(spc, mel_basis.T)))
        fbank = fbank.T
        fbank = scaler(segment(fbank, seglen=hp.seglen))
        phonation_parms = scaler(extract_phonation(the_wavpath,seglen=hp.seglen))
        feature_set = np.concatenate((fbank,phonation_parms),axis=0)
        np.save(the_npypath, feature_set)
        print("{}|{} -- processing {} -- frame_length:{} -- dimention:{}".format(count,total,
        wav.replace("\\","/").split("/")[-1],feature_set.shape[-1],feature_set.shape[0]))
    print("*"*20+"finished extracting feature Fbank and Phonation Parms"+"*"*20)

def generate_paired_index(hp:Processor_Hparams,data_dir:str="feature_MFCC",root_dir:str="CNN_BiLSTM"):
    '''
    :param hp: Processor_Hparams
    :param data_dir: npy存储的根目录
    :param root_dir: 训练索引的存储目录
    root_dir以模型名称命名（不能超出集合）
    '''
    if root_dir in ('CNN_BiLSTM', 'CNN', 'CNN_LSTM'):
        root_dir = root_dir
    else:
        raise ValueError("root dir is not in set('CNN_BiLSTM', 'CNN', 'CNN_LSTM')")
    data_dir = data_dir # npy文件存储的目录 例如feature_MFCC
    save_dir = root_dir + "/" + data_dir # 以同名文件夹存储在root_dir下
    if not os.path.exists(save_dir): # 如果存储目录不存在
        os.makedirs(save_dir) # 创建存储文件夹

    npy_path_list = [] # npy文件路径列表
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for file in filenames:  
            if os.path.splitext(file)[1] == '.npy':  # 判断文件是否是npy文件
                npy_path_list.append(os.path.join(dirpath, file)) # 将所有npy文件路径添加到列表

    ######################   训练集和验证集的划分 生成两个txt文件 ##############################
    ### 我们随机划分训练集、测试集（不划分验证集）比例自己选择
    ### 从每个分类中提取80%比例的数据进行训练,剩下20%作为测试

    ###  取出该数据集中的音频类别（子文件夹名称）
    audio_classes = []
    for dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, dir)):
            audio_classes.append(str(dir))
    ## print(audio_classes) ## ['healthy', 'pathology']

    ## 创建2个字典保存训练集和测试集路径
    ## 字典结构为 {"healthy":[路径1，路径2.] "pathology":[路径1，路径2.]}。
    train_dict = {}
    test_dict = {}
    for audio_class in audio_classes:
        train_dict[audio_class] = []
        test_dict[audio_class] = []
    ## 空字典创建完毕 按照类别写入npy路径

    ## 循环音频类别（子文件夹名称）
    ## 将文件夹内npy路径添加到字典
    for audio_class in audio_classes:
        npy_files = []
        ## 添加当前类别的npy文件路径
        for npy_file in npy_path_list:
            if npy_file.replace('\\', '/').split("/")[-2] == audio_class: ## npy文件按照audio_class分类
                ### npy_file.split("\\")[-2]代表子文件夹名字 linux下是"/" 这里做了兼容
                npy_files.append(npy_file)
        ## 按比例对不同类别的音频进行训练集和测试集的划分
        file_counts = len(npy_files)
        ## print(f"{audio_class} class contains {file_counts} files")
        cutoff = int(file_counts * hp.train_ratio) ## 90%的训练集
        random.shuffle(npy_files) ###  随机打乱
        train_paths = npy_files[:cutoff] ## 取前90%
        test_paths = npy_files[cutoff:]  ## 取后10%
        ## 将上面两个训练集和测试集列表存储到对应key值的字典
        train_dict[audio_class] += train_paths #这里 列表加列表 必须使用 +=
        test_dict[audio_class] += test_paths

    ## 将字典里的路径索引写入到文件
    with open((save_dir + "/" + "train.txt"),'w', encoding='utf-8') as f:
        for _,v in train_dict.items():
            for p in v:
                f.write(str(p) + "\n")
    with open((save_dir + "/" + "test.txt"),'w', encoding='utf-8') as f:
        for _,v in test_dict.items():
            for p in v:
                f.write(str(p) + "\n")
    print(f"*"*20 + str({data_dir}) + "训练集和测试集表单分割完毕" + "*"*20)

if __name__ == '__main__':

    '''
    从音频文件中提取特征存储为npy文件
    将特征npy切分为训练集和测试集txt
    '''

    '''
    如果想要以--形式传参，可以使用argparse模块
    可在命令行中输入python preprocess.py -i ./audios_and_mats -o ./feature_set_1
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir","-i", type=str, default="./audios_and_mats", help="directory to the wavs")
    parser.add_argument("--output_dir", "-o", type=str, default="./feature_set_1", help="directory to the outputs")
    args = parser.parse_args().__dict__
    input_dir: str = args.pop("input_dir")
    output_dir: str = args.pop("output_dir")
    preprocess_hp.set_dir(input_dir, output_dir)
    extract_feature_set1(preprocess_hp)
    '''

    ## 生成随机种子
    same_seeds(2022)
    
    ## 提取指定特征
    processor_hp = Processor_Hparams() 
    extract_feature_Melspec(processor_hp,'audios_and_mats', 'feature_Melspec')
    extract_feature_Melspec_and_Phonation(processor_hp,'audios_and_mats', 'feature_Melspec_and_Phonation')
    extract_feature_MFCC(processor_hp,'audios_and_mats', 'feature_MFCC')
    extract_feature_MFCC_and_Phonation(processor_hp,'audios_and_mats', 'feature_MFCC_and_Phonation')
    extract_feature_Fbank(processor_hp,'audios_and_mats', 'feature_Fbank')
    extract_feature_Fbank_and_Phonation(processor_hp,'audios_and_mats', 'feature_Fbank_and_Phonation')

    ## 生成训练集和测试集索引 以CNN_BiLSTM为例
    generate_paired_index(processor_hp,"feature_Melspec","CNN_BiLSTM")
    generate_paired_index(processor_hp,"feature_Melspec_and_Phonation","CNN_BiLSTM")
    generate_paired_index(processor_hp,"feature_MFCC","CNN_BiLSTM")
    generate_paired_index(processor_hp,"feature_MFCC_and_Phonation","CNN_BiLSTM")
    generate_paired_index(processor_hp,"feature_Fbank","CNN_BiLSTM")
    generate_paired_index(processor_hp,"feature_Fbank_and_Phonation","CNN_BiLSTM")
    
    ## 生成训练集和测试集索引 以CNN_LSTM为例
    generate_paired_index(processor_hp,"feature_Melspec","CNN_LSTM")
    generate_paired_index(processor_hp,"feature_Melspec_and_Phonation","CNN_LSTM")
    generate_paired_index(processor_hp,"feature_MFCC","CNN_LSTM")
    generate_paired_index(processor_hp,"feature_MFCC_and_Phonation","CNN_LSTM")
    generate_paired_index(processor_hp,"feature_Fbank","CNN_LSTM")
    generate_paired_index(processor_hp,"feature_Fbank_and_Phonation","CNN_LSTM")

    ## 生成训练集和测试集索引 以CNN为例
    generate_paired_index(processor_hp,"feature_Melspec","CNN")
    generate_paired_index(processor_hp,"feature_Melspec_and_Phonation","CNN")
    generate_paired_index(processor_hp,"feature_MFCC","CNN")
    generate_paired_index(processor_hp,"feature_MFCC_and_Phonation","CNN")
    generate_paired_index(processor_hp,"feature_Fbank","CNN")
    generate_paired_index(processor_hp,"feature_Fbank_and_Phonation","CNN")