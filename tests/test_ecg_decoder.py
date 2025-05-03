import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ecgtokenizer.QRSTokenizer import QRSTokenizer
import torch
from ecgtokenizer.tokenizer import ECGTokenizer
from ecgtokenizer.modeling_ecgdecoder import get_model_default_params, ECGDecoder

# 1つ目のECG
data1, fields1 = wfdb.rdsamp("test_wave/00001_lr")
data1 = np.array(data1).transpose(1, 0)[np.newaxis, :, :]  # shape: [1, channels, seq_len]

# 2つ目のECG（別ファイルまたは同じデータのコピーでもOK）
data2, fields2 = wfdb.rdsamp("test_wave/00001_lr")  # 実際には別のファイルを使うとよい
data2 = np.array(data2).transpose(1, 0)[np.newaxis, :, :]

# バッチにまとめる（shape: [2, channels, seq_len]）
data = np.concatenate([data1, data2], axis=0)
print(data.shape)  # torch.Size([2, channels, seq_len])

# TokenizerとModel処理
tokenizer = ECGTokenizer()
out = tokenizer(data)
for i in range(190, 230):
    out["attention_mask"][0][i] = 1
print(out["attention_mask"])
params = get_model_default_params()
model = ECGDecoder(**params)

# 1回目
out1 = model(
    out["token"],
    out["attention_mask"],
    in_chan_matrix=out["in_chan_matrix"],
    in_time_matrix=out["in_time_matrix"],
    return_qrs_tokens=False,
)
print(out1.shape, out1.sum())
out = tokenizer(data)
for i in range(190, 230):
    out["attention_mask"][0][i] = 1
print(out["attention_mask"])

# 2回目（少しトークンをカットする）
out2 = model(
    out["token"][:, :-10],
    out["attention_mask"][:, :-10],
    in_chan_matrix=out["in_chan_matrix"],
    in_time_matrix=out["in_time_matrix"],
    return_qrs_tokens=False,
)
print(out2.shape, out2.sum())
