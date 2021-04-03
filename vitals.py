import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from model import MTTS_CAN

from process import preprocess_raw_video,detrend
def hear_rate(peaklist,fs):
    RR_list = []
    cnt = 0
    
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    
    bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
    return bpm
    
img_rows = 36
img_cols = 36
frame_depth = 10
def load_model():
    model_checkpoint = './mtts_can.hdf5'
    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    return model

def predict_vitals(video_path,model):
    
    
    batch_size = 100
    sample_data_path = video_path
    distance=10
    dXsub,fs = preprocess_raw_video(sample_data_path, dim=36)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
    print(yptest)
    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(2, [0.65 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    
    return pulse_pred,resp_pred,fs