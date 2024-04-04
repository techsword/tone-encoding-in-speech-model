import glob
import os
import re

import numpy as np
import pandas as pd
import textgrids
import torch
from preprocessing import (check_dimension, read_textgrids,
                           save_textgrids_to_csvs)
# from ..embgen import save_textgrids_to_csvs
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

dataset_path = '~/data_thchs30/data'

def add_pinyin_to_df(save_file = 'dataset_insight.csv'):
    if os.path.isfile(save_file):
        df = pd.read_csv(save_file)
    else:
        df = save_textgrids_to_csvs()
        # expand the file ID to full file paths
        file_IDs = df['file_ID'].unique()
        trn_paths = pd.Series(file_IDs).map(lambda x: os.path.expanduser(os.path. join(dataset_path, x + ".wav.trn")))
        file_lookup_dict = dict(zip(file_IDs, trn_paths))
        for i in file_lookup_dict:
            assert i in file_lookup_dict[i]

        def get_pinyin(trn_file_ID):
            trn_file = file_lookup_dict[trn_file_ID]
            with open(trn_file) as f:
                trn_file_content = f.read().splitlines()
                pinyin_tokenized = trn_file_content[1]
            return pinyin_tokenized, trn_file_ID

        list_of_pinyin = [get_pinyin(trn_file_ID) for trn_file_ID in file_IDs]

        for x in tqdm(list_of_pinyin):
            try:
                df.loc[(df['file_ID'] == x[1]) & (~df['transcription'].str.contains('SIL')), 'pinyin'] = x[0].split()
            except:
                # print(len(x[0].split()), len(df.loc[(df['file_ID'] == x[1]) & (~df['transcription'].str.contains('SIL')), 'pinyin']))
                print(x[1])
                break
        df.to_csv(save_file)
    return df

def compare_sentence_lengths(x):
    ortho_len = len(''.join(x['transcription'].split()))
    pinyin_len = len((x['pinyin'].split()))
    try:
        assert ortho_len == pinyin_len
    except:
        # print(ortho_len, pinyin_len)
        print('Nope', x)

class thchsDataset(Dataset):

    def __init__(self, file_IDs, corpus_path =  '~/data_thchs30/data'):
        self.file_IDs = file_IDs
        self.corpus_path = os.path.expanduser(corpus_path)

    def __len__(self):
        return len(self.file_ID)

    def __getitem__(self, idx):
        file_ID = self.file_IDs[idx]
        trn_file = os.path.join(self.corpus_path, file_ID + ".wav.trn")
        with open(trn_file, 'r') as f:
            trns = f.read().splitlines()
        trn_chars = trns[0]
        trn_pinyin = trns[1]

        return {'pinyin': trn_pinyin, 'chars': trn_chars, 'file_ID': file_ID}

def dataset_insights(save_file = 'dataset_insight.csv'):

    if os.path.isfile(save_file):
        df = pd.read_csv(save_file, index_col=0)
    else:
        segmented_dataset = save_textgrids_to_csvs()
        file_IDs = segmented_dataset['file_ID'].unique()
        no_sil_df = segmented_dataset[~segmented_dataset.transcription.str.contains('SIL')].copy().sort_values(by=['file_ID', 'startTime'])
        df_all_transcription = pd.read_csv('thchs30_data.csv', 
                                             names=['file_ID', 'transcription', 'pinyin', 'separate_pinyin'], 
                                             header = 0).to_numpy()
        all_phonetic_splits = []

        for entry in tqdm(df_all_transcription):
            file_ID, _, phonetic_transcriptions, _ = entry
            pinyin_split = [(file_ID, x) for x in phonetic_transcriptions.split(' ')]
            all_phonetic_splits.extend(pinyin_split)

        all_phonetic_splits = np.array(all_phonetic_splits)
        filtered_phonetic_splits = all_phonetic_splits[np.isin(all_phonetic_splits[:,0],no_sil_df.file_ID.unique().astype('str'))]


        no_sil_df.loc[:,"phonetic"] = filtered_phonetic_splits[:,1]

def filter_special_consonants():
    consonants = ['r', 'sh', 'ch','s','z','j','zh','q','c','x']
    consonants_pattern = '|'.join(consonants)
    vowels = 'aeiou'
    pattern_str = f'({consonants_pattern})([{vowels}])(.*)'

    pattern = re.compile(pattern_str)
    save_file = 'dataset_insight.csv'
    df = pd.read_csv(save_file, index_col=0)
    df = df.dropna()
    df = df[~df['transcription'].str.contains('sil|SIL')]

    df.reset_index(names = 'all_index', inplace=True)
    filtered_df = df[df['pinyin'].str.contains(pattern_str)].copy()
    filtered_df.reset_index(names = 'filtered_index', inplace=True)
   
    filtered_df['pinyin_wo_tone'] = filtered_df['pinyin'].map(lambda x: x[:-1])
    filtered_df.loc[:,'onset'] = filtered_df.loc[:,'pinyin_wo_tone'].map(lambda x: re.sub(pattern_str, r'\1', x))
    filtered_df.loc[:,'endings'] = filtered_df.loc[:,'pinyin_wo_tone'].map(lambda x: re.sub(pattern_str, r'\2\3', x))
    filtered_df['tone_label'] = filtered_df['pinyin'].map(lambda x: x[-1])
    
    filtered_indices = filtered_df['filtered_index'].to_numpy()
    consonant_labels = filtered_df['onset'].to_numpy()

    return filtered_indices, consonant_labels