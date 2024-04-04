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
from torch.utils.data import Dataset
from generate_aligned_dataset import save_aligned_dataset_csv
from scipy.signal import resample
from tqdm.auto import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load audio file
dataset_path = "~/data_thchs30/data/"
dataset_path = os.path.expanduser(dataset_path)
file_ID = "A2_54.wav"
audio_file_path = os.path.join(dataset_path,file_ID)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

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
        dataset_path = "~/vivos/train/waves/"
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
    return 

def slice_extracted_audio_features(datasetname = 'thchs30',
                                     sr = 16000, 
                                     time_step = 0.01,
                                     num_sample_points = 20,
                                     num_context_ms = 10):
    audio_features_file = f'./audio_features/{datasetname}_audio_feats.pt'
    raw_audio_features = torch.load(audio_features_file)

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Start processing extracted audio features to classifier input file")

    audio_features_chainmap = ChainMap(*raw_audio_features)

    aligned_dataset = save_aligned_dataset_csv(dataset = datasetname, rewrite=False).to_numpy()
    experiment_file_IDs = np.unique(aligned_dataset[:,0])

    file_IDs, feat_list = [], []
    for wav_file in tqdm(audio_features_chainmap, desc='Slicing features'):

        file_ID = os.path.basename(wav_file).split('.')[0]
        audio_features = audio_features_chainmap[wav_file]
        if file_ID not in experiment_file_IDs:
            continue
        segment_numpified_df = aligned_dataset[aligned_dataset[:,0] == file_ID]
        def get_word_feats(word_alignment):
            word_f0_values = audio_features['praat_f0'][(word_alignment[1] < audio_features['praat_time']) & (audio_features['praat_time'] < word_alignment[2])]
            word_f0_values[word_f0_values == 0] = np.nan
            try:
                # Interpolates the nan values to make up for pitch tracking
                interpolated_f0 = np.interp(np.arange(len(word_f0_values)),
                        np.arange(len(word_f0_values))[np.isnan(word_f0_values) == False], 
                        word_f0_values[np.isnan(word_f0_values) == False])
                resampled_word_f0_values = resample(interpolated_f0, num = num_sample_points)
                word_f0_mean = np.nanmean(word_f0_values)
            except:
                # logging.warning(f"the f0 is missing values, filling with 0s")
                resampled_word_f0_values = np.zeros(10)
                word_f0_mean = np.zeros(1)

            word_mfcc_values = audio_features['librosa_mfcc'][:,(word_alignment[1] < audio_features['librosa_time']) & (audio_features['librosa_time'] < word_alignment[2])]
            word_mfcc_mean = np.nanmean(word_mfcc_values, axis = 1)
            

            word_center = (word_alignment[1] + word_alignment[2])/2
            word_center_idx = (np.abs(audio_features['praat_time'] - word_center)).argmin() # find the index of the word center in numpy array
            left,right = word_center_idx-num_context_ms, word_center_idx+num_context_ms+1
            left = 0 if left < 0 else left
            f0_ryant,mfcc_ryant = np.zeros(21), np.zeros(840)
            word_f0_values = audio_features['praat_f0'][left:right]
            word_mfcc_values = np.concatenate(audio_features['librosa_mfcc'][:, left:right])
            if left == 0:
                f0_ryant[-len(word_f0_values):] = word_f0_values
                mfcc_ryant[-len(word_mfcc_values):] = word_mfcc_values
            else:
                f0_ryant[:len(word_f0_values)] = word_f0_values
                mfcc_ryant[:len(word_mfcc_values)] = word_mfcc_values
            f0_ryant[f0_ryant == 0] = np.nan
            mfcc_ryant[mfcc_ryant == 0] = np.nan
            processed_audio_features = {'f0': f0_ryant,
                                        'mfcc': mfcc_ryant,
                                        'f0-resampled': resampled_word_f0_values,
                                        'mfcc-concat': word_mfcc_values,
                                        'f0-mean': word_f0_mean,
                                        'mfcc-mean': word_mfcc_mean}
            return processed_audio_features
        
        word_feats = [get_word_feats(word) for word in segment_numpified_df]
        feat_list.append(word_feats)
        file_IDs.append(file_ID)
    return file_IDs, feat_list

class audiofeatureDataset(Dataset):

    def __init__(self, file_IDs, feat_list,model_ID = 'f0', datasetname = 'thchs30'):
        """_summary_

        Args:
            features (_type_): _description_
            datasetname (str, optional): 'thchs30' or 'vivos'. Defaults to 'thchs30'.
        """
        self.file_IDs = file_IDs
        self.embs = feat_list
        self.model_ID = model_ID
        self.datasetname = datasetname
        self.transformed_dataset = save_aligned_dataset_csv(datasetname).to_numpy()
        consonants = ['r', 'sh', 'ch','s','z','j','zh','q','c','x']
        consonants_pattern = '|'.join(consonants)
        vowels = 'aeiou'
        self.pattern_str = f'({consonants_pattern})([{vowels}])(.*)'

    def __len__(self):
        return len(self.file_IDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_ID = self.file_IDs[idx]
        seg_emb = np.array([x[self.model_ID] for x in self.embs[idx]])
        trns = self.transformed_dataset[self.transformed_dataset[:,0] == file_ID]
        trn_chars = trns[:,3].astype(str)
        split_phonetic = trns[:,4].astype(str)
        phonetic_wo_tone = np.array([x[:-1] for x in split_phonetic])
        tone_labels = np.array([int(x[-1]) if x[-1].isdigit() else 0 for x in split_phonetic])

        onset = np.array(list(map(lambda x: re.sub(self.pattern_str, r'\1', x), phonetic_wo_tone)))
        ending = np.array(list(map(lambda x: re.sub(self.pattern_str, r'\2\3', x), phonetic_wo_tone)))

        return {'model_ID':self.model_ID,
                'datasetname':self.datasetname,
                'embs': seg_emb, 
                'chars': trn_chars, 
                'tone_labels': tone_labels,
                'onset': onset,
                'ending': ending,
                'split_phonetic' : split_phonetic, 
                'phonetic_wo_tone':phonetic_wo_tone, 
                'file_ID': file_ID}

def save_to_classifier_input(file_IDs, feat_list, 
                             datasetname = 'thchs30', 
                             save_path = 'classifier_input'):
    feat_names = ['f0','mfcc','f0-resampled','mfcc-mean'] #'mfcc-concat','f0-mean'
    for model_ID in feat_names:
        save_name = os.path.join(save_path, f"{model_ID}_{datasetname}_extracted-data.pt")

        dataset = audiofeatureDataset(file_IDs, feat_list, model_ID=model_ID, datasetname=datasetname)
        dataset_ready_to_save = list(dataset)
        torch.save(dataset_ready_to_save, save_name, pickle_protocol = 4)
        

def main():
    generated_classifier_input_save_path = 'classifier_input'
    for datasetname in ['thchs30', 'vivos-train']:
        audio_features_file = run_extract_audio_features(datasetname = datasetname, save_path = 'audio_features', flattened = False, parallel = True)
        existing_files = glob.glob(generated_classifier_input_save_path + f"/*f0*{datasetname}*.pt")
        if len(existing_files) != 0:
            print(f'{datasetname} has saved audio features like {existing_files} already')
            continue
        file_IDs, feat_list = slice_extracted_audio_features(datasetname = datasetname,
                                        sr = 16000, 
                                        time_step = 0.01,
                                        num_sample_points = 10)
        save_to_classifier_input(file_IDs, feat_list, 
                                datasetname = datasetname, 
                                save_path = generated_classifier_input_save_path)




if __name__ == "__main__":
    main()
