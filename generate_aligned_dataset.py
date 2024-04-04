import glob
import os
import re

import numpy as np
import pandas as pd
import textgrids
from tqdm.auto import tqdm


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
        transcription = '[SIL]' if len(transcription) == 0 else transcription
        tg_out.append((file_ID, startTime, endTime, transcription))
    return tg_out

def read_vphon_out(file_vphone_out = '~/vivos/train/prompts_ipa.txt'):
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
                             rewrite = False):
    if 'vivos' in dataset.lower():
        alignment_path = '~/vivos/alignment_train/waves'
    elif 'thchs' in dataset.lower():
        alignment_path = '~/data_thchs30/thchs-aligned'
    else:
        raise KeyError(f"{dataset} isn't legal")
    
    save_csv = f"{dataset}_aligned.csv"
    transformed_dataset = f'{dataset}_transformed_dataset.csv'

    if os.path.isfile(save_csv) and not rewrite:
        no_sil_df = pd.read_csv(transformed_dataset)
    else:
        absolute_tg_files = glob.glob(os.path.expanduser(alignment_path) + "/**/*.TextGrid", recursive=True)
        processed_textgrids = [np.array(read_textgrids(tg_file)) for tg_file in tqdm(absolute_tg_files, desc='Reading textgrids')]

        if 'vivos' in alignment_path.lower():
            vphon_out = read_vphon_out().to_numpy()
            def get_phonetic(processed_textgrid):
                trns = vphon_out[vphon_out[:,0] == processed_textgrid[0,0]]
                phonetic_transcriptions = trns[0,1].split()
                len_transcription = len(phonetic_transcriptions)
                len_textgrid_alignment = processed_textgrid[np.where(processed_textgrid[:,-1] != '[SIL]')].shape[0]
                return len_transcription == len_textgrid_alignment, phonetic_transcriptions
        elif 'thchs' in alignment_path.lower():
            def get_phonetic(tg_out, dataset_path = '~/data_thchs30/data', trn_ending = ".wav.trn"):
                tg_out = np.array(tg_out)
                trn_file = os.path.join(os.path.expanduser(dataset_path), tg_out[0,0] + trn_ending)
                with open(trn_file, 'r') as f:
                    trns = f.read().splitlines()
                phonetic_transcriptions = trns[1].split()
                len_transcription = len(phonetic_transcriptions)
                len_textgrid_alignment = tg_out[np.where(tg_out[:,-1] != '[SIL]')].shape[0]
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
        no_sil_df.to_csv(transformed_dataset, index = None)

    return no_sil_df


def main():
    thchs_df = save_aligned_dataset_csv(dataset = 'thchs30', 
                       rewrite = True)
    vivos_df = save_aligned_dataset_csv(dataset = 'vivos-train',
                             rewrite = True)

if __name__ == "__main__":
    main()