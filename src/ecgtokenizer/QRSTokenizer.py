# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# ---------------------------------------------------------
from wfdb import processing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


class QRSTokenizer(nn.Module):
    def __init__(self, fs, max_len, token_len, used_chans=None):
        super(QRSTokenizer, self).__init__()
        print(
            'Make sure that ECG Leads are sorted by ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]'
        )
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.used_chans = used_chans

    def qrs_detection(self, ecg_signal):
        channels, _ = ecg_signal.shape  # [C, signal value]
        all_qrs_inds = []

        for channel_index in range(channels):
            lead_signal = ecg_signal[channel_index, :]
            qrs_inds = processing.xqrs_detect(
                sig=lead_signal, fs=self.fs, verbose=False
            )
            all_qrs_inds.append(qrs_inds)
        return all_qrs_inds

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []

        for channel_index in range(channels):
            qrs_segments = []
            for i in range(len(qrs_inds[channel_index])):
                if i == 0:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    if (i + 1) < len(qrs_inds[channel_index]):
                        end = end = (
                            qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                        ) // 2
                    else:
                        end = min(
                            start + self.token_len, len(ecg_signal[channel_index])
                        )
                elif i == len(qrs_inds[channel_index]) - 1:
                    center = qrs_inds[channel_index][i]
                    end = min(
                        center + self.token_len // 2, len(ecg_signal[channel_index])
                    )
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                else:
                    center = qrs_inds[channel_index][i]
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                    end = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                    ) // 2

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len

                # print(f"Segment start: {segment_start}, Segment end: {segment_end}")

                segment[segment_start:segment_end] = ecg_signal[channel_index][
                    start:end
                ]

                qrs_segments.append(segment)

            channel_qrs_segments.append(qrs_segments)

        return channel_qrs_segments

    def assign_time_blocks(self, qrs_inds, interval_length=100):
        in_time = [(ind // interval_length) + 1 for ind in qrs_inds]
        return in_time

    def qrs_to_sequence(self, channel_qrs_segments, qrs_inds):
        qrs_sequence = []
        in_chans = []
        in_times = []

        for channal_index, channel in enumerate(channel_qrs_segments):
            in_times.extend(self.assign_time_blocks(qrs_inds[channal_index]))
            for segments in channel:
                qrs_sequence.append(segments)
                # in_chans.append(channal_index + 1)
                in_chans.append(self.used_chans[channal_index] + 1)

        current_patch_size = len(qrs_sequence)
        if current_patch_size < self.max_len:
            padding_needed = self.max_len - current_patch_size
            for _ in range(padding_needed):
                qrs_sequence.append(np.zeros(self.token_len))
                in_chans.append(0)
                in_times.append(0)

        elif current_patch_size > self.max_len:
            qrs_sequence = qrs_sequence[: self.max_len]
            in_chans = in_chans[: self.max_len]
            in_times = in_times[: self.max_len]

        return np.stack(qrs_sequence), np.array(in_chans), np.array(in_times)

    def forward(self, x):
        r"""
        x (batch, channels, n_samples): ecg_wave
        """
        x = x[:, self.used_chans, :]
        bs, c, l = x.shape
        batch_qrs_seq = []
        batch_in_chans = []
        batch_in_times = []

        for batch in range(bs):
            ecg_signal = x[batch]
            qrs_inds = self.qrs_detection(ecg_signal)
            channel_qrs_segments = self.extract_qrs_segments(ecg_signal, qrs_inds)

            qrs_sequence, in_chans, in_times = self.qrs_to_sequence(
                channel_qrs_segments, qrs_inds
            )

            batch_qrs_seq.append(qrs_sequence)
            batch_in_chans.append(in_chans)
            batch_in_times.append(in_times)

        batch_qrs_seq = np.array(batch_qrs_seq).astype(np.float32)
        batch_in_chans = np.array(batch_in_chans).astype(np.int64)
        batch_in_times = np.array(batch_in_times).astype(np.int64)

        return batch_qrs_seq, batch_in_chans, batch_in_times
