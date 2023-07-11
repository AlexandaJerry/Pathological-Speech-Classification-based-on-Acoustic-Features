# 病变语音识别：基于传统声学参数和发声参数的深度学习

├── CNN  
│   ├── feature_Fbank  
│   ├── feature_Fbank_and_Phonation  
│   ├── feature_MFCC  
│   ├── feature_MFCC_and_Phonation  
│   ├── feature_Melspec  
│   └── feature_Melspec_and_Phonation  
├── CNN_BiLSTM  
│   ├── feature_Fbank  
│   ├── feature_Fbank_and_Phonation  
│   ├── feature_MFCC  
│   ├── feature_MFCC_and_Phonation  
│   ├── feature_Melspec  
│   └── feature_Melspec_and_Phonation  
├── CNN_LSTM  
│   ├── feature_Fbank  
│   ├── feature_Fbank_and_Phonation  
│   ├── feature_MFCC  
│   ├── feature_MFCC_and_Phonation  
│   ├── feature_Melspec  
│   └── feature_Melspec_and_Phonation  
├── Model.py  
├── Mydataset.py  
├── README.md  
├── Trainer.py  
├── audios_and_mats  
│   ├── healthy  
│   └── pathology  
├── auto_runner.py  
├── feature_Fbank  
├── feature_Fbank_and_Phonation  
├── feature_MFCC  
├── feature_MFCC_and_Phonation  
├── feature_Melspec  
├── feature_Melspec_and_Phonation  
├── hparms.py  
├── main.py  
├── network_visualization  
├── network_visualization.ipynb  
├── processor.py  
├── requirements.txt  
├── tree.md  
├── utils.py  
├── 自然语言处理课堂汇报-Mou.ppt  
└── 自然语言处理课程论文-Mou.pdf    



1. 想直接进行训练和测试

```
python main.py
```

```
python auto_runner.py
```

2. 想重新提取参数和做数据集

```
python processor.py
```

   

![1](../picture/1.png)

![2](../picture/2.png)

![3](../picture/3.png)
