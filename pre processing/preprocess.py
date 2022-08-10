from concurrent.futures import process
# from types import NoneType
import numpy as np 
import pandas as pd 
import os 
import cv2
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
from scipy import integrate
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt 


sampling_frequency = 30

def bandPassFilter(signal):
    fs = 30
    lowcut = 0.7
    highcut = 4

    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    order = 3

    b, a = butter(order, [low,high], "bandpass", analog = False)
    y = filtfilt(b, a, signal,axis = 0)
    
    return(y)

def get_normalized_signal(filtered_ppg):
    filtered_ppg = np.array(filtered_ppg).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler = scaler.fit(filtered_ppg)
    normalized = scaler.transform(filtered_ppg)
    # normalized = normalize(filtered_ppg)
    return normalized

def get_clean_signal(ppg_signal):

    ppg_signal = np.array(ppg_signal)
    # remove nan values from the signal 
    # for i in range(ppg_signal.size):
    #     if np.isnan(ppg_signal[0]):
    #         ppg_signal = np.delete(ppg_signal, 0)
    
    # # if entire signal consists of Nan values, discard the signal
    # if ppg_signal.size < 21:
    #     return 'NaN signal'
    
    # filter the signal and normalize amplitudes 
    filtered_ppg = bandPassFilter(ppg_signal)
    normalized = get_normalized_signal(filtered_ppg)
    normalized = normalized.squeeze()
    return normalized


def systolic_peaks(signal):
    ''' Returns list of found systolic peaks in whole signal. Identified as maxima. Required distance between peaks set to 22'''
    return find_peaks(signal, distance=20)[0]

def tfn_points(signal):
    ''' Returns list of tfn points (beats boundaries) as minimums of signal with distance min 25 between eachother'''
    
    # here I use reverted signal and get peaks above 0 
    return find_peaks(signal*(-1), distance=20)[0]

def beat_segmentation(signal):
    
    ''' Returns list of beats from signal and list of corresponding systolic peak index'''
    
    systolics = systolic_peaks(signal)
    tfns = tfn_points(signal)
    
    beats, systolic = [], []
    
    for i in range(len(tfns)-1):
        start = tfns[i]
        end = tfns[i+1]
        segment = np.arange(start, end)
        l = [f in systolics for f in segment]
        
        # if there is only one systolic peak between minima its a beat
        if list(map(bool, l)).count(True) == 1: 
            # apply normalization, reshaping is required
            bshape = signal[segment].shape
            normalized_beat = normalize(signal[segment].reshape(1, -1))
            beats.append(normalized_beat.reshape(bshape))
            systolic.append(np.where(l)[0][0])
            
    
    return beats, systolic

def dicrotic_notch(beat, systolic):
    '''Returns index of detected dicrotic notch in a beat. If not found returns 0'''
    
    derviative = np.diff(np.diff(beat[systolic:]))
    point = find_peaks(derviative)[0]
    corrected = 0
    
    if len(point) > 0:
        corrected =  systolic + point[-1]
        
    return corrected

def diastolic_peak(beat, systolic):
    '''Returns index of detected diastolic peak in a beat. If not found returns 0'''
   
    derviative = np.diff(np.diff(beat[systolic:]))
    point = find_peaks(derviative*(-1))[0]
    corrected = 0
    
    if len(point) > 0:
        corrected = systolic + point[-1]
        if abs(beat[corrected]) >= abs(1.01*beat[corrected - 1]):
            return corrected
        else: return 0
        
    return corrected

def peaks_detection(beats, systolics):
    '''Returns created dataframe with beat values and critical points indices'''
    
    dicrotics = []
    diastolics = []
    
    for b, s in zip(beats, systolics):
        tnn = dicrotic_notch(b,s)
        tdn = diastolic_peak(b,s)
        
        dicrotics.append(tnn)
        diastolics.append(tdn)
    
    result = np.array([beats, systolics, dicrotics, diastolics], dtype=object)
    # remove those where dicrotics and diastolics weren't found
    result = result[..., result[2] > 0]
    result = result[..., result[3] > 0]
    
    # output shape is (4, nb) where nb is number of beats
    return result.T


################ FEATURES #####################
def heart_rate(signal, fs):
    ''' Number of systolic peaks per minute. Normal between 60-100 bpm'''
    
    sys = systolic_peaks(signal)
    T = len(signal)/fs
    
    return len(sys)/(T/60)

def reflection_index(beat, systolic, diastolic):
    ''' Returns reflection index which is systolic amplitude to systolic-diastolic amplitude ratio'''
    
    a = beat[systolic] - np.min(beat)
    b = a - (beat[diastolic] - np.min(beat))
    
    return a/b

def systolic_timespan(dicrotic, fs):
    ''' Returns systolic beat timespan in seconds (dicrotic peak marks its end)'''
    
    return dicrotic/fs

def up_time(systolic, fs):
    ''' Returns time to systolic peak in seconds '''
    
    return systolic/fs
    
def systolic_volume(beat, dicrotic, fs):
    ''' Returns systolic volume as area under the curve '''
    
    return integrate.simps(beat[:dicrotic], dx=1/fs)
    
def diastolic_volume(beat, dicrotic, fs):
    ''' Returns diastolic volume as area under the curve '''
    
    return integrate.simps(beat[dicrotic:], dx=1/fs)

def extract_features(signal, beats_arr, fs):
    '''Extract features from beat array consisting of beat values, systolic peak index, dicrotic notch index and diastolic peak index'''
    
    features = []    
    hr = heart_rate(signal, fs)
    
    for beat, systolic, dicrotic, diastolic in beats_arr:
        
        ri = reflection_index(beat, systolic, diastolic)
        st = systolic_timespan(dicrotic, fs)
        ut = up_time(systolic, fs)
        sv = systolic_volume(beat, dicrotic, fs)
        dv = diastolic_volume(beat, dicrotic, fs)
        features.append([hr, ri, st, ut, sv, dv])
    features = np.mean(features, axis=0)
    # return shape should be (Tx, 6)
    return np.array(features)

def process_signal(data, fit=True):
    ''' Process single signal function, returns normlized features after filtering signal and beat segmentation '''
        
    filtered = get_clean_signal(data)
    # if type(filtered) == str:
    #     return False 
    beats, sys = beat_segmentation(filtered)
    peaks = peaks_detection(beats, sys)
    features = extract_features(filtered, peaks, sampling_frequency)

    # if fit:
    #     scaler = scaler.partial_fit(features)
    #     features = scaler.transform(features)
    # print(features.shape)
    # if features.shape == (0,):
    #     return 'None'
    return features 

def prepare_dataset(data, store_address, fit=True):
    ''' Perform processing all signals in the dataset '''
    x = np.zeros((1611, 6))
    for i in range(len(data)):
        a = process_signal(data[i], fit)
        if np.isnan(a):
            continue 
        else:
            x[i] = a
    # x = np.array([process_signal(sample, count, fit) for sample in data for count in range(187)], dtype=object)
    x = pd.DataFrame(x)
    os.chdir(store_address)
    x.to_csv('final_dataset_x_newaug.csv')
    return x


