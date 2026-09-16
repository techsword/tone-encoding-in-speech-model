import glob
import os
import re

import numpy as np
import pandas as pd
import textgrids
from tqdm.auto import tqdm


# ---- Corpus locations -----------------------------------------------------
# Point THCHS30/VIVOS at your local copies. Defaults are repo-relative; set
# the environment variables to your own layout (see README "Data
# requirements" for the paths used in the paper).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORPORA_ROOT = os.environ.get('CORPORA_ROOT', os.path.join(_REPO_ROOT, 'corpora'))
THCHS30_DIR = os.environ.get('THCHS30_DIR', os.path.join(CORPORA_ROOT, 'data_thchs30'))
VIVOS_DIR = os.environ.get('VIVOS_DIR', os.path.join(CORPORA_ROOT, 'vivos'))


def read_textgrids(filename):
    if 'thchs' in filename.lower():
        speaker_ID, utt_ID = os.path.basename(filename).replace('.TextGrid','').replace('TH','').split('-')
        file_ID = '_'.join([re.sub(r'(?<=\D)0(?=\d)','',speaker_ID),re.sub(r'^0+(?=\d)','', utt_ID)])
    elif 'vivos' in filename.lower():
        file_ID = os.path.basename(filename).replace('.TextGrid', '')

    grid = textgrids.TextGrid(filename)
    word_invervals = grid['words']
    tg_out = []
    for word in word_invervals:
        startTime = word.xmin
        endTime = word.xmax
        transcription = word.text
        tg_out.append((file_ID, startTime, endTime, transcription))
    return tg_out

def check_dimension(tg_out, dataset_path = None, trn_ending = ".wav.trn"):
    if dataset_path is None:
        dataset_path = os.path.join(THCHS30_DIR, 'data')
    tg_out = np.array(tg_out)
    trn_file = os.path.join(os.path.expanduser(dataset_path), tg_out[0,0] + trn_ending)
    with open(trn_file, 'r') as f:
        trns = f.read().splitlines()
    phonetic_transcriptions = trns[1].split()
    len_transcription = len(phonetic_transcriptions)
    len_textgrid_alignment = tg_out[np.where(tg_out[:,-1] != '[SIL]')].shape[0]
    return len_transcription == len_textgrid_alignment, phonetic_transcriptions

def read_vphon_out(file_vphone_out = None):
    if file_vphone_out is None:
        file_vphone_out = os.path.join(VIVOS_DIR, 'train', 'prompts_ipa.txt')
    file = os.path.expanduser(file_vphone_out)
    with open(file, 'r') as f:
        prompts_ipa = f.readlines()
    ipa_, ortho_ = [], []
    for x in prompts_ipa:
        if '[' in x:
            ipa_.append(re.sub(r"\[|\]", "",x).strip().upper().split(' ', 1))
        else:
            ortho_.append(x.strip().split(' ', 1))
    ipa_df = pd.DataFrame(ipa_, columns= ['fileid', 'ipa'])
    ortho_df = pd.DataFrame(ortho_, columns= ['fileid', 'ortho'])
    df = pd.merge(ipa_df, ortho_df)
    return df

def thchs_save_dataset(thchs_aligned_path = None, 
                       save_csv = 'thchs-aligned-filtered.csv', 
                       rewrite = False):
    if thchs_aligned_path is None:
        thchs_aligned_path = os.path.join(THCHS30_DIR, 'thchs-aligned')
    if os.path.isfile(save_csv) and not rewrite:
        df = pd.read_csv(save_csv)
    else:
        absolute_tg_files = glob.glob(os.path.expanduser(thchs_aligned_path)+"/*.TextGrid")
        processed_textgrids = [np.array(read_textgrids(tg_file)) for tg_file in tqdm(absolute_tg_files, desc='Reading textgrids')]
        dimension_conformity_mask, phonetic_transcriptions = list(zip(*[check_dimension(x) for x in tqdm(processed_textgrids, desc='Dimension check')]))
        dimension_conformity_mask = np.array(dimension_conformity_mask)
        phonetic_transcriptions = np.concatenate(np.array(phonetic_transcriptions, dtype = object)[dimension_conformity_mask])
        cleaned_processed_textgrids = np.concatenate(np.array(processed_textgrids, dtype = object)[dimension_conformity_mask])
        
        df = pd.DataFrame(cleaned_processed_textgrids,columns = ['file_ID','startTime', 'endTime', 'transcription'])
        df.to_csv(save_csv, index=None)

        no_sil_df = df[~df.transcription.str.contains('sil|SIL')].copy()
        no_sil_df['phonetic_transcriptions'] = phonetic_transcriptions
        no_sil_df['tone_label'] = no_sil_df['phonetic_transcriptions'].map(lambda x: x[-1])
        no_sil_df.to_csv('thchs30_transformed_dataset.csv')
    return df

def save_textgrids_to_csvs(thchs_aligned_path = None,
                           save_csv = 'thchs-aligned-filtered.csv',
                           rewrite = False):
    """Read THCHS-30 TextGrids and return the aligned dataset as a DataFrame.

    Kept for compatibility with `dataset_insights`; identical to
    `thchs_save_dataset`.
    """
    return thchs_save_dataset(thchs_aligned_path = thchs_aligned_path,
                              save_csv = save_csv,
                              rewrite = rewrite)

def vivos_save_dataset(save_csv = 'vivos_train_aligned.csv', 
                           rewrite = False, 
                           alignment_path = None):
    if alignment_path is None:
        alignment_path = os.path.join(VIVOS_DIR, 'alignment_train', 'waves')


    if os.path.isfile(save_csv) and not rewrite:
        df = pd.read_csv(save_csv)
    else:
        absolute_tg_files = glob.glob(os.path.join(os.path.expanduser(alignment_path), "**/*.TextGrid"), recursive=True)
        processed_textgrids = [np.array(read_textgrids(tg_file)) for tg_file in tqdm(absolute_tg_files, desc='Reading textgrids')]
        vphon_out = read_vphon_out().to_numpy()
        def get_phonetic(processed_textgrid):
            trns = vphon_out[vphon_out[:,0] == processed_textgrid[0,0]]
            phonetic_transcriptions = trns[0,1].split()
            len_transcription = len(phonetic_transcriptions)
            len_textgrid_alignment = len(processed_textgrid[~(processed_textgrid[:,-1] == 'SIL')])
            return len_transcription == len_textgrid_alignment, phonetic_transcriptions
        dimension_conformity_mask, phonetic_transcriptions = list(zip(*[get_phonetic(x) for x in tqdm(processed_textgrids, desc='Dimension check')]))

        dimension_conformity_mask = np.array(dimension_conformity_mask)
        phonetic_transcriptions = np.concatenate(np.array(phonetic_transcriptions, dtype = object)[dimension_conformity_mask])
        cleaned_processed_textgrids = np.concatenate(np.array(processed_textgrids, dtype = object)[dimension_conformity_mask])
        
        df = pd.DataFrame(cleaned_processed_textgrids,columns = ['file_ID','startTime', 'endTime', 'transcription'])

        df.to_csv(save_csv, index=None)

        no_sil_df = df[~df.transcription.str.contains('sil|SIL')].copy()
        no_sil_df['phonetic_transcriptions'] = phonetic_transcriptions
        no_sil_df['tone_label'] = no_sil_df['phonetic_transcriptions'].map(lambda x: x[-1])
        no_sil_df.to_csv('vivos_transformed_dataset.csv')

    return df

def main():
    thchs_df = thchs_save_dataset(thchs_aligned_path = os.path.join(THCHS30_DIR, 'thchs-aligned'), 
                       save_csv = 'thchs-aligned-filtered.csv', 
                       rewrite = True)
    vivos_df = vivos_save_dataset(save_csv = 'vivos_train_aligned.csv', 
                           rewrite = False, 
                           alignment_path = os.path.join(VIVOS_DIR, 'alignment_train', 'waves'))
    pass

if __name__ == "__main__":
    main()