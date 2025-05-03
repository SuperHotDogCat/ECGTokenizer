import os
import requests
from tqdm import tqdm  # プログレスバー用
from typing import Optional
import torch
import torch.nn as nn
from ecgtokenizer.QRSTokenizer import QRSTokenizer
from ecgtokenizer.modeling_vqhbr import get_model_default_params, vqhbr, VQHBR


class ECGTokenizer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        qrs_config: Optional[dict] = None,
        vqvhr_config: Optional[dict] = None,
        ignore_index: int = 15,
        add_cls_to_attention_mask: bool = True,
    ):
        super().__init__()
        if not model_path and os.path.isfile("checkpoint-100-mimiciv.pth"):
            model_path = "checkpoint-100-mimiciv.pth"
        if not model_path:
            url = "https://huggingface.co/PKUDigitalHealth/HeartLang/resolve/main/checkpoints/vqhbr/MIMIC-IV/checkpoint-100.pth?download=true"
            output_filename = "checkpoint-100-mimiciv.pth"
            with requests.get(url, stream=True) as response:
                if response.status_code == 200:
                    total_size = int(response.headers.get("content-length", 0))
                    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
                    with open(output_filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                    print(f"Saved as {output_filename}")
                    progress_bar.close()
                else:
                    print(f"Failed to download: {response.status_code}")
            model_path = "checkpoint-100-mimiciv.pth"
        if not qrs_config:
            qrs_config = dict(
                fs=100, max_len=256, token_len=96, used_chans=[i for i in range(12)]
            )
        qrs_tokenizer = QRSTokenizer(**qrs_config).eval()

        if not vqvhr_config:
            vqhbr_model = vqhbr().eval()
        else:
            vqhbr_model = VQHBR(**vqvhr_config).eval()
        param = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        vqhbr_model.load_state_dict(param["model"])

        self.qrs_tokenizer = qrs_tokenizer.eval()
        self.vqhbr_model = vqhbr_model.eval()
        self.ignore_index = ignore_index
        self.add_cls_to_attention_mask = add_cls_to_attention_mask
        self.all_reduce_fn = lambda x: None # ddpの対象にはしない

    def __call__(self, ecg):
        batch_qrs_seq, batch_in_chans, batch_in_times = self.qrs_tokenizer(ecg)
        batch_qrs_seq = torch.tensor(batch_qrs_seq)
        batch_in_chans = torch.tensor(batch_in_chans)
        batch_in_times = torch.tensor(batch_in_times)
        out = self.vqhbr_model.get_tokens(batch_qrs_seq, batch_in_chans, batch_in_times)
        out["in_chan_matrix"] = batch_in_chans
        out["in_time_matrix"] = batch_in_times
        attention_mask = (out["token"] != self.ignore_index).int()
        if self.add_cls_to_attention_mask:
            ones = torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([ones, attention_mask], dim=1)
        out["ecg_attention_mask"] = attention_mask # 一応attentionの学習をよくするために追加, ダメなら消す
        return out
