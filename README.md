# 病变语音识别：基于传统声学参数和发声参数的深度学习
 
├── CNN
│   ├── feature_Fbank
│   ├── feature_Fbank_and_Phonation
│   ├── feature_MFCC
│   ├── feature_MFCC_and_Phonation
│   ├── feature_Melspec
│   └── feature_Melspec_and_Phonation
├── CNN_BiLSTM
│   ├── feature_Fbank
│   ├── feature_Fbank_and_Phonation
│   ├── feature_MFCC
│   ├── feature_MFCC_and_Phonation
│   ├── feature_Melspec
│   └── feature_Melspec_and_Phonation
├── CNN_LSTM
│   ├── feature_Fbank
│   ├── feature_Fbank_and_Phonation
│   ├── feature_MFCC
│   ├── feature_MFCC_and_Phonation
│   ├── feature_Melspec
│   └── feature_Melspec_and_Phonation
├── Model.py
├── Mydataset.py
├── README.md
├── Trainer.py
├── audios_and_mats
│   ├── healthy
│   └── pathology
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

表1 发声参数汇总
Phonation Parameters
基频扰动	Jitter	第一第二谐波差值	H1-H2
振幅扰动	Shimmer	第二第四谐波差值	H2-H4
谐波噪声比	HNR	第一谐波与第一共振峰近邻谐波差值	H1-A1
倒谱峰值显度	CPP	第一谐波与第二共振峰近邻谐波差值	H1-A2
声门激励强度	SOE	第一谐波与第三共振峰近邻谐波差值	H1-A3
