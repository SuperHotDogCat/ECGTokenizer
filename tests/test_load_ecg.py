import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ecgtokenizer.QRSTokenizer import QRSTokenizer
import torch
from ecgtokenizer.modeling_vqhbr import vqhbr
from ecgtokenizer.tokenizer import ECGTokenizer

data, fields = wfdb.rdsamp("test_wave/40689238")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :]
print(data.shape)
print(data.mean())
