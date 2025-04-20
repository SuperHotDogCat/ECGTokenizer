import numpy as np
import wfdb
import matplotlib.pyplot as plt
from ecgtokenizer.QRSTokenizer import QRSTokenizer
import torch
from ecgtokenizer.modeling_vqhbr import vqhbr


data, fields = wfdb.rdsamp("test_wave/00001_lr")
data = np.array(data).transpose(1, 0)[np.newaxis, :, :]
print(data, fields)
print(data.shape)
tokenizer = QRSTokenizer(
                fs=100,
                max_len=256, # ECGを何tokenとるか(idのsequence長)
                token_len=96, # 1 tokenのサンプル数
                used_chans=[i for i in range(12)],
            )
batch_qrs_seq, batch_in_chans, batch_in_times = tokenizer(data)
batch_qrs_seq = torch.tensor(batch_qrs_seq)
batch_in_chans = torch.tensor(batch_in_chans)
batch_in_times = torch.tensor(batch_in_times)

vqhbr_model = vqhbr()
param = torch.load("../checkpoint-100.pth", map_location=torch.device('cpu'), weights_only=False)
vqhbr_model.load_state_dict(param["model"])

out = vqhbr_model.get_tokens(batch_qrs_seq, batch_in_chans, batch_in_times)
print(out["token"])
print("token shape: ", out["token"].shape, "token type: ", out["token"].dtype)
