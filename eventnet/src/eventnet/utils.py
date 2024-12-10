import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy as sp
import os

from eventnet.architecture import UNet

def load_model():
    model = UNet(input_channels=19)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(os.path.join(dir_path, 'model.pth'), weights_only=True))
    return model

class SeizureDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size_sec=120, fs=256):
        self.eeg = data
        self.fs = fs
        self.window_size = int(fs*window_size_sec)
        self.max_duration = int(fs*60*5)
        self.recording_duration = int(data.shape[1] / fs)

        self.pad = 256
        window_idx = np.arange(self.pad, self.recording_duration*self.fs, self.window_size).astype(int)
        self.window_idx = window_idx[window_idx < self.recording_duration*self.fs - self.window_size]

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        eeg_clip = self.eeg[:, self.window_idx[idx]-self.pad//2:self.window_idx[idx]+self.window_size+self.pad//2]
        x = torch.FloatTensor(eeg_clip)
        return x

def custom_collate_fn(batch):
    x = torch.cat([item for item in batch if item is not None], dim=0)
    return x

def get_dataloader(data, window_size_sec=120, fs=256):
    dataset = SeizureDataset(data=data, window_size_sec=window_size_sec, fs=fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, n_workers=4)
    return dataloader

def events_to_mask(output_centers, output_duration, window_size_sec, fs=256, thresh=0.5):
    window_size = int(window_size_sec * fs)
    centers = output_centers
    durations = output_duration
    max_duration = fs*60*5

    mask = np.zeros(centers.shape)
    for i in range(centers.shape[0]):
        centers_smooth = gaussian_filter1d(centers[i, :], 100)
        durations_smooth = durations[i, :]
        peaks = sp.signal.find_peaks(centers_smooth, height=thresh, distance=fs*60)[0]  # at least 1 minute distance between peaks
        pred_event = []
        
        for peak in peaks:
            pred_event.append((peak, durations_smooth[peak] * max_duration))
        for center, duration in pred_event:
            if center - duration//2 < 0:
                diff = abs(center - duration//2)
                mask[max(0, i-1), -int(diff):] = 1 # add to previous window
            if center + duration//2 > window_size:
                diff = center + duration//2 - window_size
                mask[min(i+1, mask.shape[0]-1), :int(diff)] = 1
            mask[i, int(max(0, center-duration//2)):int(min(window_size, center+duration//2))] = 1
    return mask

def predict(model, dataloader, device, window_size_sec, recording_duration, fs, thresh):
    pred_centers = []
    pred_durations = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            output_centers, output_durations = model(data)
            output_centers = output_centers.squeeze()
            output_durations = output_durations.squeeze()

            pred_centers.append(output_centers.cpu().detach().numpy())
            pred_durations.append(output_durations.cpu().detach().numpy())

    pred_centers = np.array(pred_centers)  # Concat
    pred_durations = np.array(pred_durations)

    pred_mask = events_to_mask(pred_centers, pred_durations, window_size_sec, fs=fs, thresh=thresh)
    preds = np.concatenate(pred_mask)

    y_predict = np.zeros(int(recording_duration*fs))
    y_predict[:len(preds)] = preds

    return y_predict

