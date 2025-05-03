import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ecgtokenizer.QRSTokenizer import QRSTokenizer
import torch
from ecgtokenizer.modeling_vqhbr import vqhbr
from ecgtokenizer.tokenizer import ECGTokenizer

data, fields = wfdb.rdsamp("test_wave/00001_lr")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :]
print(data, fields)
print(data.shape)
tokenizer = ECGTokenizer()
out = tokenizer(data)
print(out)

data, fields = wfdb.rdsamp("test_wave/40689238")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :1500] # 今の構造だと1500 sequenceまでしか対応できないので,　今後訓練することがあったらやる
print(data, fields)
print(data.shape)
tokenizer = ECGTokenizer()
out = tokenizer(data)
print(out)
