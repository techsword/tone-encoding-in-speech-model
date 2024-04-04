import glob
import logging
import math
import os
from collections import ChainMap
import re

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import torch
from preprocessing import thchs_save_dataset
from scipy.signal import resample
from tqdm.auto import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load audio file
dataset_path = "~/data_thchs30/data/"
dataset_path = os.path.expanduser(dataset_path)
file_ID = "A2_54.wav"
audio_file_path = os.path.join(dataset_path,file_ID)

def extract_audio_features(audio_file_path:str | os.PathLike):
    pitch_floor=60
    pitch_ceiling=650
    wintime = 0.025
    hoptime = 0.010
    nbands = 40
    numcep = 40
    minfreq = 0
    maxfreq = 8000
    dcttype = 3


    # Extract MFCC features with Librosa
    y,sr = librosa.load(audio_file_path, sr= None)
    hop_len = int(sr*hoptime)
    win_len = int(sr*wintime)
    y_mfccs = librosa.feature.mfcc(y=y, 
                                   sr=sr, 
                                   n_mfcc=numcep, 
                                   n_fft=win_len, 
                                   hop_length=hop_len, 
                                   n_mels=nbands,
                                   fmin=minfreq, 
                                   fmax=maxfreq, 
                                   dct_type=dcttype)
    
    librosa_time = librosa.times_like(y_mfccs, sr = sr, hop_length = hop_len, n_fft = win_len)

    snd = parselmouth.Sound(values = y, sampling_frequency = sr)
    # Extract f0 contour with parselmouth/praat
    pitch = snd.to_pitch(time_step = hoptime, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    praat_f0 = pitch.selected_array['frequency']
    praat_time = pitch.xs()

    return {os.path.basename(audio_file_path):{'praat_f0':praat_f0, 
                                               'praat_time': praat_time,
                                               'librosa_mfcc':y_mfccs, 
                                               'librosa_time':librosa_time,
                                            #    'librosa_f0':y_f0,
                                               }}

def run_extract_audio_features(datasetname = 'thchs30', save_path = 'audio_features', flattened = False, parallel = True):

    if 'thchs30' in datasetname:
        dataset_path = "~/data_thchs30/data/"
        dataset_path = os.path.expanduser(dataset_path)
    elif 'vivos' in datasetname:
        dataset_path = "~/vivos/train/waves"
        dataset_path = os.path.expanduser(dataset_path)
        datasetname = 'vivos-train'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    all_audio = [file for file in glob.glob(dataset_path+"**/*.wav" , recursive=True) if 'flat' not in file]
    savename = os.path.join(save_path, f'{datasetname}_audio_feats.pt')
    if os.path.isfile(savename):
        print(f'{savename} exists already! skipping')
    else:
        if parallel:
            out = joblib.Parallel(n_jobs=18, verbose=1)(
                joblib.delayed(extract_audio_features)(i) for i in tqdm(all_audio[:])
            )
        else:
            out = [extract_audio_features(x) for x in tqdm(all_audio)]
        torch.save(out, savename)
    return savename

def process_extracted_audio_features(audio_features_file = './audio_features/thchs30_audio_feats.pt',
                                     datasetname = 'thchs30',
                                     sr = 16000, 
                                     time_step = 0.01,
                                     num_sample_points = 20):
    raw_audio_features = torch.load(audio_features_file)

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Start processing extracted audio features to classifier input file")

    audio_features_chainmap = ChainMap(*raw_audio_features)

    df = thchs_save_dataset(rewrite=False)
    numpified_df = df.to_numpy()
    file_ids_in_experiment = np.unique(numpified_df[:,0])
    all_transcription = pd.read_csv('thchs30_data.csv', names=['file_ID', 'orthography', 'pinyin', 'pinyin_split']).to_numpy()

    all_labels,word_f0_contours_mean, word_mfccs_mean, word_f0_contours, word_mfccs = [],[],[],[],[]
    word_f0_contours_ryant, word_mfccs_ryant = [], []
    num_context_ms = 10
    context_frame_sec = num_context_ms * time_step
    extension = '.wav'
    if 'thchs30' in datasetname:
        dataset_path = "~/data_thchs30/"
        dataset_path = os.path.expanduser(dataset_path)
    elif 'vivos' in datasetname:
        dataset_path = "~/vivos/train/"
        dataset_path = os.path.expanduser(dataset_path)
        datasetname = 'vivos-train'
    glob_list = [x for x in glob.glob(dataset_path + '/**/*' + extension, recursive = True) if 'flat' not in x]
    tg_list = [x for x in glob.glob(dataset_path + '/**/*' + '.TextGrid', recursive = True)]

    #TODO: make this function prettier and more efficient
    for tg_file in tqdm(tg_list):
        speaker_ID, utt_ID = os.path.basename(tg_file).replace('.TextGrid','').replace('TH','').split('-')
        file_ID = '_'.join([re.sub(r'(?<=\D)0(?=\d)','',speaker_ID),re.sub(r'^0+(?=\d)','', utt_ID)])
        wav_file = file_ID.replace('TH','').replace('-','_')+'.wav'
        audio_features = audio_features_chainmap[wav_file]
        if file_ID not in file_ids_in_experiment:
            continue
        segment_numpified_df = numpified_df[numpified_df[:,0] == file_ID]
        segment_numpified_df = segment_numpified_df[np.where(segment_numpified_df[:,3] != '[SIL]')]

        split_pinyins = all_transcription[all_transcription[:,0]==file_ID][:,-2].item().split()
        all_labels.extend([int(x[-1]) for x in split_pinyins])

        for word_alignment in segment_numpified_df:
            word_f0_values = audio_features['praat_f0'][(word_alignment[1] < audio_features['praat_time']) & (audio_features['praat_time'] < word_alignment[2])]
            word_f0_values[word_f0_values == 0] = np.nan
            try:
                # Interpolates the nan values to make up for pitch tracking
                interpolated_f0 = np.interp(np.arange(len(word_f0_values)),
                        np.arange(len(word_f0_values))[np.isnan(word_f0_values) == False], 
                        word_f0_values[np.isnan(word_f0_values) == False])
                resampled_word_f0_values = resample(interpolated_f0, num = num_sample_points)
            except:
                # logging.warning(f"the f0 is missing values, filling with 0s")
                resampled_word_f0_values = np.zeros(10)

            word_mfcc_values = audio_features['librosa_mfcc'][:,(word_alignment[1] < audio_features['librosa_time']) & (audio_features['librosa_time'] < word_alignment[2])]
            word_mfcc = np.nanmean(word_mfcc_values, axis = 1)

            

            word_f0_contours_mean.append(resampled_word_f0_values)
            word_mfccs_mean.append(word_mfcc)
            word_f0_contours.append(word_f0_values)
            word_mfccs.append(word_mfcc_values)

            word_center = (word_alignment[1] + word_alignment[2])/2
            word_center_idx = (np.abs(audio_features['praat_time'] - word_center)).argmin() # find the index of the word center in numpy array
            left,right = word_center_idx-num_context_ms, word_center_idx+num_context_ms+1
            left = 0 if left < 0 else left
            f0_container,mfcc_container = np.zeros(21), np.zeros(840)
            word_f0_values = audio_features['praat_f0'][left:right]
            word_mfcc_values = np.concatenate(audio_features['librosa_mfcc'][:, left:right])
            if left == 0:
                f0_container[-len(word_f0_values):] = word_f0_values
                mfcc_container[-len(word_mfcc_values):] = word_mfcc_values
            else:
                f0_container[:len(word_f0_values)] = word_f0_values
                mfcc_container[:len(word_mfcc_values)] = word_mfcc_values
            f0_container[f0_container == 0] = np.nan
            mfcc_container[mfcc_container == 0] = np.nan

            word_f0_contours_ryant.append(f0_container)
            word_mfccs_ryant.append(mfcc_container)
    
    
    all_labels_array = np.array(all_labels)
    processed_audio_features = {'f0': np.array(word_f0_contours_mean),
                                'mfcc': np.array(word_mfccs_mean),
                                'f0-concat': word_f0_contours,
                                'mfcc-concat': word_mfccs,
                                'f0-ryant': np.vstack(word_f0_contours_ryant),
                                'mfcc-ryant': np.vstack(word_mfccs_ryant)}
    for audio_feature_name in processed_audio_features.keys():
        audio_features = processed_audio_features[audio_feature_name]
        if 'concat' in audio_feature_name:
            max_len_inputs = max([x.shape[-1] for x in audio_features])
            padded_arrays = [pad_along_axis(row, max_len_inputs, -1) for row in audio_features]
            padded_arrays = [np.concatenate(x) if len(x.shape)>1 else x for x in padded_arrays ]
            audio_features = np.vstack(padded_arrays)
        all_inputs_array = np.expand_dims(audio_features, 1)
        input_shape = all_inputs_array.shape
        if len(input_shape) !=3:
            all_inputs_array = np.expand_dims(all_inputs_array, axis = 1)
        assert len(all_inputs_array) == len(all_labels_array)
        savename = f'data/{audio_feature_name}_thchs30_extracted-data.pt'
        logging.info(f'Saving to {savename}')
        torch.save((audio_feature_name, datasetname, all_inputs_array, all_labels_array), savename, pickle_protocol = 4)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def mfcc_generation(all_audio):
    mfcc_dict = {}
    for audio in tqdm(all_audio[:]):
        y,sr = librosa.load(audio)
        y_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        mfcc_dict[os.path.basename(audio)] = y_mfccs
    return mfcc_dict


def f0_job(audio):
    y,sr = librosa.load(audio)
    f0, voiced_flag, voiced_probs = librosa.pyin(y,fmin=50,fmax=500)
    return os.path.basename(audio), f0



def generate_audio_features(datasetname = 'thchs30', save_path = 'audio_features', flattened = False, parallel = True):
    """_summary_

    Args:
        datasetname (str, optional): _description_. Defaults to 'thchs30'.
        save_path (str, optional): _description_. Defaults to 'audio_features'.
        flattened (bool, optional): _description_. Defaults to False.
        parallel (bool, optional): _description_. Defaults to True.
    """
    if 'thchs30' in datasetname:
        dataset_path = "~/data_thchs30/data/"
        dataset_path = os.path.expanduser(dataset_path)

    elif 'vivos' in datasetname:
        dataset_path = "~/vivos/train/waves"
        dataset_path = os.path.expanduser(dataset_path)
        datasetname = 'vivos-train'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    all_audio = [file for file in glob.glob(dataset_path+"**/*.wav" , recursive=True) if 'flat' not in file]
    
    mfcc_savename = os.path.join(save_path, f'{datasetname}_mfccs.pt')
    if os.path.isfile(mfcc_savename):
        print(f"{mfcc_savename} exists, not overwriting")
    else:
        print(f"generating mfcc to {mfcc_savename}")
        mfcc_dict = mfcc_generation(all_audio)
        torch.save(mfcc_dict, mfcc_savename)


    f0_savename = os.path.join(save_path, f'{datasetname}_f0s.pt')
    if os.path.isfile(f0_savename):
        print(f"{f0_savename} exists, not overwriting")
    else:
        print(f"generating mfcc to {f0_savename}")
        if parallel:
            import joblib
            jobs = [ joblib.delayed(f0_job)(i) for i in tqdm(all_audio[:]) ]
            out = joblib.Parallel(n_jobs=8, verbose=1)(jobs)
        else:
            out = [f0_job(x) for x in tqdm(all_audio)]

        f0_dict = dict(out)
        torch.save(f0_dict, f0_savename)

def process_data(datasetname = 'thchs30', save_path = 'audio_features/', audio_feature = 'f0', flattened = False, parallel = True,aggregate_method = 'avgpool'):
    """_summary_

    Args:
        datasetname (str, optional): _description_. Defaults to 'thchs30'.
        save_path (str, optional): _description_. Defaults to 'audio_features/'.
        audio_feature (str, optional): _description_. Defaults to 'f0'.
        flattened (bool, optional): _description_. Defaults to False.
        parallel (bool, optional): _description_. Defaults to True.
        aggregate_method (str, optional): _description_. Defaults to 'avgpool'.
    """
    if 'thchs30' in datasetname:
        dataset_path = "~/data_thchs30/data/"
        dataset_path = os.path.expanduser(dataset_path)
        file = [x for x in glob.glob(save_path+'*.pt') if datasetname in x and audio_feature in x and 'processed' not in x]
        assert len(file) == 1
        raw_dataset = torch.load(file[0])
        all_audio = [file for file in glob.glob(dataset_path+"**/*.wav" , recursive=True) if 'flat' not in file]
        _, sr = librosa.load(all_audio[0])
        df = thchs_save_dataset(rewrite=False)


    elif 'vivos' in datasetname:
        dataset_path = "~/vivos/train/waves"
        dataset_path = os.path.expanduser(dataset_path)
        datasetname = 'vivos-train'


    all_segment_list, labels_list = [], []
    numpified_df = df.to_numpy()

    for file_ID in tqdm(raw_dataset.keys()):
        segment_numpified_df = numpified_df[numpified_df[:,0] == file_ID.split('.')[0]]
        if len(segment_numpified_df) == 0:
            continue
        segment_numpified_df = segment_numpified_df[np.where(segment_numpified_df[:,3] != '[SIL]')]
        trn_file = os.path.join(os.path.expanduser(dataset_path), file_ID + ".trn")
        with open(trn_file, 'r') as f:
            trns = f.read().splitlines()
        pinyin_transcriptions = trns[1].split()
        assert len(segment_numpified_df) == len(pinyin_transcriptions)
        timestamps = librosa.times_like(raw_dataset[file_ID], sr = sr)
        audio_features = raw_dataset[file_ID]
        if len(audio_features.shape) != 1:
            audio_features = np.moveaxis(audio_features, -1, 0)
        segments = []

        for word in segment_numpified_df:
            segment = audio_features[(word[1] < timestamps) & (timestamps < word[2])]
            segments.append(segment)

        max_len = max([x.shape[0] for x in segments])
        segments_array = np.array([pad_along_axis(row, max_len) for row in segments]) 
        labels = np.array([int(x[-1]) for x in pinyin_transcriptions])
        all_segment_list.append(segments_array)
        labels_list.append(labels)

    

    all_labels_array = np.concatenate(labels_list)
    max_len_inputs = max([x.shape[1] for x in all_segment_list])
    all_inputs_array = np.concatenate([pad_along_axis(row, max_len_inputs, 1) for row in all_segment_list])
    if aggregate_method == 'avgpool':
        all_inputs_array[all_inputs_array == 0] = np.nan
        means = np.nanmean(all_inputs_array, axis=1)
        all_inputs_array = np.nan_to_num(means, copy=True, nan=0.0, posinf=None, neginf=None)



    assert len(all_labels_array) == len(all_inputs_array)

    torch.save({'inputs': all_inputs_array, 'labels': all_labels_array},
               f'/home/gshen/work_dir/speech-model-tone-probe/audio_features/{datasetname}_{audio_feature}_processed.pt')



    all_inputs_array = np.expand_dims(all_inputs_array, 1)
    input_shape = all_inputs_array.shape
    if len(input_shape) !=3:
        all_inputs_array = np.expand_dims(all_inputs_array, axis = 1)

    data_savename = f'data/{audio_feature}_{datasetname}_extracted-data.pt'
    torch.save((audio_feature, datasetname, all_inputs_array, all_labels_array), data_savename)

def main():
    audio_features_file = run_extract_audio_features(datasetname = 'thchs30', save_path = 'audio_features', flattened = False, parallel = True)
    process_extracted_audio_features(audio_features_file = audio_features_file,
                                     datasetname = 'thchs30',
                                     sr = 16000, 
                                     time_step = 0.01,
                                     num_sample_points = 10)




if __name__ == "__main__":
    main()
