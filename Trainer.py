import torch
import os.path
import torch.nn as nn
import pickle ## 帮助保存参数
from sklearn.metrics import accuracy_score ###  帮助算分类准确率
from torch.utils.data import DataLoader  ### 帮助创建数据集
from torchsummary import summary ## 打印模型结构
from Mydataset import MydataSet  ## 数据集类
from hparms import Training_Hparams  ## 参数类
from Model import CNN_BiLSTM, CNN_LSTM, CNN ## 模型类

class Trainer(object):

    def __init__(self, hp: Training_Hparams):
        super(Trainer, self).__init__()
        self.hp = hp
        self.device = hp.device
        self.set_trainer_configuration()
        pass

    def set_trainer_configuration(self):
        ## 训练前的一系列初始化
        self.early_stop = False  ## 是否提前停止训练
        self.epoch_num = 0 ## 在迭代第几次数据集
        self.current_iter = 0  ## 当前在迭代第几次batch
        self.init_logger()    ## 创建 日志文件
        self.prepareDataloader()  ## 创建数据集
        self.build_models()  ## 创建模型
        pass

    def init_logger(self):
        '''
        删除已经存在的日志文件
        log_train_{version}.txt
        log_eval_{version}.txt
        存储超参数到hparams_{version}.txt
        '''
        self.log_train_filepath = str(self.hp.log_train_filepath)
        self.log_eval_filepath = str(self.hp.log_eval_filepath)
        ## 如果已经存在log_train_{version}.txt文件就删除重新创建
        if os.path.exists(self.hp.log_train_filepath):
            os.remove(self.hp.log_train_filepath)  ## 删除已经存在的日志文件
        ## 如果已经存在log_eval_{version}.txt文件就删除重新创建
        if os.path.exists(self.hp.log_eval_filepath):
            os.remove(self.hp.log_eval_filepath)  ## 删除已经存在的日志文件
        ## 存储超参数
        with open(self.hp.hp_filepath, 'w', encoding='utf-8') as wf:
            for k, v in self.hp.__dict__.items():
                wf.write("{} : {}\n".format(k, v))
            wf.write('-' * 30 + "Hparams saved" + "-" * 30 + "\n")
            wf.write("*" * 100 + "\n")
            wf.close()

    def prepareDataloader(self):
        '''
        创建数据集
        MydataSet.__len__()  返回数据集的长度
        MydataSet.__getitem__()  返回数据集中第index个数据的特征和标签
        MydataSet.__init__()  获得数据集的npy特征路径汇总和总类别数
        DataLoader.batch_size  每个batch的大小
        DataLoader.shuffle  是否打乱数据集
        DataLoader.drop_last  是否丢弃最后一个不足batch_size的batch
        '''
        ## train data set
        self.Train_dataSet = MydataSet(version_dir=self.hp.version_dir,txt_name="train.txt")
        self.TrainDataNum = self.Train_dataSet.__len__()
        self.TraindataLoader = DataLoader(self.Train_dataSet,
                                        batch_size=self.hp.batchsize,
                                        shuffle=True,
                                        drop_last=True)
        ## test data set ##
        self.Test_dataSet = MydataSet(version_dir=self.hp.version_dir,txt_name="test.txt")
        self.TestDataNum = self.Test_dataSet.__len__()
        self.TestdataLoader = DataLoader(self.Test_dataSet,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=True)

    def build_models(self):

        if self.hp.model == "CNN_BiLSTM":
            self.model = CNN_BiLSTM(input_size=self.hp.n_features,hidden_size=self.hp.hidden_size,
                                num_layers=self.hp.num_layers,num_classes=self.hp.num_classes) 
                                ## hidden_size和num_layers是lstm的参数，默认2层，hidden_size是128
        # 后续补全新的模型 注意修改模型的输入输出
        elif self.hp.model == "CNN":
            self.model = CNN(input_size=self.hp.n_features,num_classes=self.hp.num_classes)
        elif self.hp.model == "CNN_LSTM":
            self.model = CNN_LSTM(input_size=self.hp.n_features,hidden_size=self.hp.hidden_size,
                                num_layers=self.hp.num_layers,num_classes=self.hp.num_classes)
        else:
            raise ValueError("model is not in set('CNN_BiLSTM', 'CNN', 'CNN_LSTM')")
        self.current_lr = self.hp.lr_start
        self.loss_func = nn.CrossEntropyLoss() ## 分类最常用的交叉熵概率
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.current_lr,amsgrad=self.hp.amsgrad,
                                betas=(self.hp.beta1, self.hp.beta2)) 
                                ## Adam优化器参数设置
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.hp.lr_dacay_every,
                                gamma = self.hp.gamma) ## 学习率衰减步数和衰减系数
        self.model_summary(self.model) ## 打印模型信息     
        self.model.to(self.hp.device) ## 将模型放到gpu

    def model_summary(self, model):
        """Print out the network information."""
        a = torch.rand(32,1,self.hp.n_features,self.hp.frame_length)
        cata = a.permute(0,1,3,2)
        summary(model, cata)

    def start_trainer(self):
        """
        Main training loop
        """
        print('------------- BEGIN TRAINING LOOP ----------------')
        while self.early_stop == False:
            self.train_by_step()
            if self.early_stop == True:
                break
    
    def train_by_step(self):
        '''
        pytorch普适的训练代码基本框架
        ## 定义数据集
        dataset = None
        dataloader = DataLoader(dataset,batch_size=5)
        ## 定义模型、优化器、损失函数
        model  = Classifier()
        lossfunction = None
        optimizer = None
        scheduler = None
        ## 定义训练循环
        for batch in dataloader:
            inputs,labels = batch
            outputs = model(inputs)
            loss = lossfunction(inputs,outputs)
            optimizer.zero_grad()   ## 计算图梯度清空
            loss.backward()   ## 利用损失值，计算每个参数的梯度
            optimizer.step() ###  利用梯度，更新网络参数
            scheduler.step() ## 学习率衰减
        '''
        for batch in self.TraindataLoader:
            self.check_stop() ## 检查是否需要停止训练
            if self.early_stop: ## 如果需要停止训练，直接跳出循环
                break
            self.current_iter += 1 ## 当前迭代次数
            self.epoch_num = self.current_iter // (self.TrainDataNum // self.hp.batchsize) ## 当前迭代的epoch数
            npy,labels = [b.to(self.device) for b in batch] # npy:[B:32,1,feature:40,frame:100] labels:[B:32]
            npy = npy.unsqueeze(1).permute(0,1,3,2) ## [B:32,1,frame:100,feature:40]
            pred_prob = self.model(npy) ## 输出分类概率 [B,num_class]
            batch_loss = self.loss_func(pred_prob,labels)        
            pred_index = torch.max(pred_prob, dim=1)[1]  # 求最大概率的下标作为预测的类别
            batch_accuracy = accuracy_score(labels.cpu(), pred_index.cpu()) ## 计算准确率
            self.optimizer.zero_grad()       ## 清空计算图中的梯度
            batch_loss.backward()    ## 计算计算图每个参数的梯度
            self.optimizer.step()    ## 更新优化器参数
            self.scheduler.step()    ## 更新学习率
            self.current_lr = self.optimizer.param_groups[0]['lr'] ## 更新学习率
            # print(self.current_lr) ## 打印学习率
            losse_curves  = {"train_step--":"",
                             "epoch":self.epoch_num,
                             "steps":self.current_iter,
                            "lr":self.current_lr,
                            "loss":batch_loss.item(),
                             "acc":batch_accuracy,
                             }
            ## 在训练第一步完成后，建立字典保存losses {"loss name” : [loss_data ]}
            if self.current_iter == 1:
                print("create loss dict")
                self.loss_log_dict = {}
                for k, v in losse_curves.items():
                    self.loss_log_dict[k] = []
                print("loss dict created")
            #######  loss save ######################################################
            for k, v in self.loss_log_dict.items():
                self.loss_log_dict[k].append(losse_curves[k])  # 把每batch的loss数据加入到 loss curves中
            strp = ''
            with open (self.hp.log_train_filepath, 'a', encoding='utf-8') as f:
                for key, value in losse_curves.items():
                    witem = '{}'.format(key) + ':{},'.format(value)
                    strp += witem
                f.write(strp)
                f.write('\n')
                print(strp)
        ########################################### 其他东西 ##########################################
            if (self.current_iter) % self.hp.eval_every == 0:
                self.test_accuracy()  ## 每eval_every步验证准确率 调用test_accuracy函数
            if (self.current_iter) % self.hp.model_save_every == 0:
                self.save_model(self.current_iter) ## 每model_save_every步保存模型 调用save_model函数
        pass

    def check_stop(self):
        '''
        判断训练是否应该停止
        '''
        ## 中断训练然后保存模型
        ## 保存loss词典为pickle文件
        if self.current_iter == self.hp.total_iters:
            self.save_model(self.current_iter)
            pickle_saving_path = str(self.hp.version_dir + "/" + 'running_loss_{}.pickle'.format(self.hp.version))
            with open(pickle_saving_path, 'wb') as f1:
                pickle.dump(self.loss_log_dict, f1)
            print("*************** Training End *******************")
            self.early_stop = True

    def test_accuracy(self):
        '''
        计算测试集的准确率
        '''
        ###  使用“test.txt"中的数据计算准确率。
        with torch.no_grad(): ## 测试的过程中不需要计算梯度。
            batch_acc_num = 0
            for batch in self.TestdataLoader:
                npy,labels = [b.to(self.device) for b in batch]   # mels:[B,80,256] labels:[B]
                npy = npy.unsqueeze(1).permute(0, 1, 3, 2)
                pred_prob = self.model(npy)  ## 输出分类概率 [B,num_class]
                batch_loss = self.loss_func(pred_prob, labels)
                pred_index = torch.max(pred_prob, dim=1)[1]  # 求最大概率的下标
                if pred_index == labels: ## 判断预测结果是否正确
                    batch_acc_num += 1 ## 正确则判对数量+1(因为测试时batch_size=1所以可以这样写)
                ## batch_acc_num = (pred_index == labels).sum() ## 原作者是这么写的但是我怀疑没法累加
            batch_accuracy = batch_acc_num / self.TestDataNum ## 准确率 = 判对数量 / 总test语音数量

            losse_curves  = {"test_step--":"",
                            "epoch":self.epoch_num,
                             "steps":self.current_iter,
                             "lr":self.current_lr,
                            "loss":batch_loss.item(),
                             "acc":batch_accuracy}
            strp = ''
            with open (self.hp.log_eval_filepath, 'a', encoding='utf-8') as f:
                for key, value in losse_curves.items():
                    witem = '{}'.format(key) + ':{},'.format(value)
                    strp += witem
                f.write(strp)
                f.write('\n')
                # print(strp) ## 验证准确率时不打印

    def save_model(self, i):
        '''
        model.state_dict() 保存模型参数
        hp.model_savedir 保存模型的路径
        {:06}.pth 保存模型的名字
        '''
        pdict = {"model":self.model.state_dict(),}
        path = self.hp.model_savedir + "/" + "{:06}.pth".format(i)
        torch.save(pdict, str(path))
        print("---------------- model saved as {:06}.pth ------------------- ".format(i))

if __name__ == "__main__":
    hp = Training_Hparams(feature_set="feature_MFCC", n_features=40)
    trainer = Trainer(hp)# 传入参数Training_Hparams初始化Trainer 
    print(hp.n_features) # 注意修改特征维度
    trainer.set_trainer_configuration() 
    trainer.start_trainer() # 开始训练

