import glob
import os
import re

import numpy as np
import pandas as pd
import textgrids
from tqdm.auto import tqdm


# ---- Corpus locations -----------------------------------------------------
# Point THCHS30/VIVOS/Yoruba at your local copies. Defaults are
# repo-relative; set the environment variables to your own layout (see
# README "Data requirements" for the paths used in the paper).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORPORA_ROOT = os.environ.get('CORPORA_ROOT', os.path.join(_REPO_ROOT, 'corpora'))
THCHS30_DIR = os.environ.get('THCHS30_DIR', os.path.join(CORPORA_ROOT, 'data_thchs30'))
VIVOS_DIR = os.environ.get('VIVOS_DIR', os.path.join(CORPORA_ROOT, 'vivos'))
YORUBA_DIR = os.environ.get('YORUBA_DIR', os.path.join(CORPORA_ROOT, 'yoruba'))


def read_textgrids(filename, tiername = 'words'):
    if 'thchs' in filename.lower():
        speaker_ID, utt_ID = os.path.basename(filename).replace('.TextGrid','').replace('TH','').split('-')
        file_ID = '_'.join([re.sub(r'(?<=\D)0(?=\d)','',speaker_ID),re.sub(r'^0+(?=\d)','', utt_ID)])
    # elif 'vivos' in filename.lower():
    else:
        file_ID = os.path.basename(filename).replace('.TextGrid', '')\
            

    grid = textgrids.TextGrid(filename)
    interval_tier = grid[tiername]
    tg_out = []
    for item in interval_tier:
        startTime = item.xmin
        endTime = item.xmax
        transcription = item.text
        transcription = '[SIL]' if len(transcription) == 0 else transcription
        tg_out.append((file_ID, startTime, endTime, transcription))
    return tg_out

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

def save_aligned_dataset_csv(dataset = 'thchs30',
                             rewrite = False,
                             tiername = None):
    if 'vivos' in dataset.lower():
        alignment_path = os.path.join(VIVOS_DIR, 'alignment_train', 'waves')
    elif 'thchs' in dataset.lower():
        alignment_path = os.path.join(THCHS30_DIR, 'thchs-aligned')
    elif 'yor' in dataset.lower():
        alignment_path = os.path.join(YORUBA_DIR, 'yor_aligned', 'aligned')
        tiername = 'phones' if not tiername else tiername
    else:
        raise KeyError(f"{dataset} isn't legal")
    
    save_prefix = f"{dataset}_{tiername}" if tiername else dataset
    tiername = tiername if tiername else "words"
    save_csv = f"{save_prefix}_aligned.csv"
    transformed_dataset = f'{save_prefix}_transformed_dataset.csv'

    if os.path.isfile(save_csv) and not rewrite:
        no_sil_df = pd.read_csv(transformed_dataset)
    else:
        absolute_tg_files = glob.glob(os.path.expanduser(alignment_path) + "/**/*.TextGrid", recursive=True)
        processed_textgrids = [np.array(read_textgrids(tg_file, tiername)) for tg_file in tqdm(absolute_tg_files, desc='Reading textgrids')]

        if 'vivos' in alignment_path.lower():
            vphon_out = read_vphon_out().to_numpy()
            def get_phonetic(processed_textgrid):
                trns = vphon_out[vphon_out[:,0] == processed_textgrid[0,0]]
                phonetic_transcriptions = trns[0,1].split()
                len_transcription = len(phonetic_transcriptions)
                len_textgrid_alignment = processed_textgrid[np.where(processed_textgrid[:,-1] != '[SIL]')].shape[0]
                return len_transcription == len_textgrid_alignment, phonetic_transcriptions
        elif 'thchs' in alignment_path.lower():
            def get_phonetic(tg_out, dataset_path = None, trn_ending = ".wav.trn"):
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
        elif 'yoruba' in alignment_path.lower():
            dataset_path = os.path.join(YORUBA_DIR, 'data')
            trn_ending = ".txt"
            all_transcription = glob.glob(os.path.expanduser(dataset_path + "**/*/*" +  trn_ending))
            def get_phonetic(tg_out):
                tg_out = np.array(tg_out)
                file_ID = tg_out[0,0]
                trn_file = [x for x in all_transcription if file_ID in x][0]

                with open(trn_file, 'r') as f:
                    trns = f.read().splitlines()
                phonetic_transcriptions = trns[0].split()
                len_transcription = len(phonetic_transcriptions)
                len_textgrid_alignment = tg_out[np.where(tg_out[:,-1] != '[SIL]')].shape[0]
                return len_transcription == len_textgrid_alignment, phonetic_transcriptions

        if tiername != 'phones':        
            dimension_conformity_mask, phonetic_transcriptions = list(zip(*[get_phonetic(x) for x in tqdm(processed_textgrids, desc='Dimension check')]))

            dimension_conformity_mask = np.array(dimension_conformity_mask)
            phonetic_transcriptions = np.concatenate(np.array(phonetic_transcriptions, dtype = object)[dimension_conformity_mask])
            cleaned_processed_textgrids = np.concatenate(np.array(processed_textgrids, dtype = object)[dimension_conformity_mask])
        else:
            cleaned_processed_textgrids =  np.concatenate(processed_textgrids)
            phonetic_transcriptions = None
            phonetic_transcriptions = cleaned_processed_textgrids[:,-1]
            phonetic_transcriptions = phonetic_transcriptions[phonetic_transcriptions != '[SIL]']


        df = pd.DataFrame(cleaned_processed_textgrids,columns = ['file_ID','startTime', 'endTime', 'transcription'])
        df.to_csv(save_csv, index=None)

        no_sil_df = df[~df.transcription.str.contains(r'\[SIL\]')].copy()
        no_sil_df['phonetic_transcriptions'] = phonetic_transcriptions

        if 'yor' not in alignment_path.lower():
            # no_sil_df['tone_label'] = no_sil_df['phonetic_transcriptions'].map(lambda x: x[-1])
            no_sil_df['tone_label'] = pd.to_numeric(no_sil_df['phonetic_transcriptions'].map(lambda x: x[-1]), errors = 'coerce')
            no_sil_df = no_sil_df.dropna()
            no_sil_df.loc[:,'tone_label'] = no_sil_df['tone_label'].astype(int)
            

        else:
            # Yoruba vowels (including those with diacritics)
            yoruba_vowels = set('aáàeéèiíìoóòuúùẹẹ́ẹ̀ọọ́ọ̀')
            # Function to check if a string contains any Yoruba vowels
            def contains_yoruba_vowel(text):
                return any(char.lower() in yoruba_vowels for char in str(text))
            def extract_yoruba_tone_from_phone(text):
                if 'H' in text:
                    return 'H'
                elif 'L' in text:
                    return 'L'
                else:
                    return 'M'  # Assume mid tone if no tone is specified
            no_sil_df = no_sil_df[no_sil_df['transcription'].apply(contains_yoruba_vowel)]
            no_sil_df.loc[:,'tone_label'] = no_sil_df['phonetic_transcriptions'].map(extract_yoruba_tone_from_phone)

        no_sil_df.to_csv(transformed_dataset, index = None)

    return no_sil_df


def main():
    thchs_df = save_aligned_dataset_csv(dataset = 'thchs30', 
                       rewrite = True)
    vivos_df = save_aligned_dataset_csv(dataset = 'vivos-train',
                             rewrite = True)

if __name__ == "__main__":
    main()