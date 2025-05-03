import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ecgtokenizer.QRSTokenizer import QRSTokenizer


data, fields = wfdb.rdsamp("test_wave/00001_lr")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :]
print(data, fields)
print(data.shape)
tokenizer = QRSTokenizer(
    fs=100,
    max_len=256,  # ECGを何tokenとるか(idのsequence長)
    token_len=96,  # 1 tokenのサンプル数
    used_chans=[i for i in range(12)],
)
batch_qrs_seq, batch_in_chans, batch_in_times = tokenizer(data)
print(batch_qrs_seq, batch_in_chans, batch_in_times)
for i in range(10, 20):
    plt.plot(batch_qrs_seq[0, i])
    plt.show()
    plt.clf()

data, fields = wfdb.rdsamp("test_wave/40689238")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :]
batch_qrs_seq, batch_in_chans, batch_in_times = tokenizer(data)
print(batch_qrs_seq, batch_in_chans, batch_in_times)
for i in range(0, 10):
    plt.plot(batch_qrs_seq[0, i])
    plt.show()
    plt.clf()
