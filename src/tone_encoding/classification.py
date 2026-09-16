import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def custom_train_test_split(X, y, mask_array, seed):
    """Makes train test split work for us

    Args:
        X (np.array): _description_
        y (np.array): _description_
        mask_array (np.array or None): boolean array for the test samples
    """
    if len(mask_array) != len(X):
        if len(mask_array) == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
        else:
            raise ValueError(f"using all data instead of heldout because mask array len: {len(mask_array)} != X len {len(X)}\n if this is not the correct behavior please check configurations")
    else:
        X_train = X[np.invert(mask_array)]
        X_test = X[mask_array]  
        y_train = y[np.invert(mask_array)]
        y_test = y[mask_array]

        if len(np.unique(y_test)) == 5:
        # Create a boolean mask for samples with label 5
            mask_train = (y_train.astype(int) != 5)
            mask_test = (y_test.astype(int) != 5)

            # Use the boolean mask to filter the data
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
    return X_train, X_test, y_train, y_test

def classification_pipeline(X: np.ndarray, 
                            y: np.ndarray, 
                            seed = 42,
                            mask_array = np.zeros(1),
                            tgt_layers = None):
    """_summary_

    Args:
        X (np.ndarray): array of size (num_samples, layer, features...)
        y (np.ndarray): array of size (num_samples,)
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """

    all_results = []
    if tgt_layers != None:
        layers = [tgt_layers]
        tqdm.write(f"Only running classifier on layers {layers}")
    else:
        layers = range(X.shape[1])

    for layer in tqdm(layers, desc='Layers'):

        X_ = np.nan_to_num(X[:,layer,:])
        X_train, X_test, y_train, y_test = custom_train_test_split(X_, 
                                                                   y, 
                                                                   mask_array = mask_array, 
                                                                   seed = seed)
        print(f"{len(y_train), len(y_test)}")
        clf =  make_pipeline(StandardScaler(with_mean=False), 
                         RidgeClassifierCV(alphas = [10 ** n for n in range(-4,2)], cv = 5))
        tqdm.write(f'running ridge classifier on layer {layer}')
        clf.fit(X_train, y_train)
        clf_alpha = clf[1].alpha_
        acc_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test,y_pred,)
        f1 = f1_score(y_test,y_pred, average='weighted')
        results = {
            'layer': layer,
            'acc_score': acc_score,
            'f1_score': f1,
            'model_alpha':clf_alpha,
            'cm': {'array': cm,
                   'labels':clf[1].classes_}
        }   
        all_results.append(results)
    return all_results

# Parsed dataset-insight frames, keyed by (path, filter_consonant). The frame
# is shared across calls; callers copy it before any mutation.
_DATASET_INSIGHT_CACHE = {}

def read_dataset_insight(dataset_insight_path = './thchs30_transformed_dataset.csv', 
                         filter_consonant = False):
    cache_key = (os.fspath(dataset_insight_path), bool(filter_consonant))
    cached_df = _DATASET_INSIGHT_CACHE.get(cache_key)
    if cached_df is not None:
        return cached_df

    df = pd.read_csv(dataset_insight_path, index_col=0)
    df = df.dropna()
    df = df[~df['transcription'].str.contains('sil|SIL')]
    phonetic_column_name = 'phonetic_transcriptions' if 'transformed_dataset' in dataset_insight_path else 'pinyin'
    df['phonetic_wo_tone'] = df[phonetic_column_name].map(lambda x: x[:-1])
    df['tone_label'] = df[phonetic_column_name].map(lambda x: x[-1])
    df.reset_index(names = 'all_index', inplace=True)
    if filter_consonant:
        consonants = ['r', 'sh', 'ch','s','z','j','zh','q','c','x']
        consonants_pattern = '|'.join(consonants)
        vowels = 'aeiou'
        pattern_str = f'({consonants_pattern})([{vowels}])(.*)'

        filtered_df = df[df[phonetic_column_name].str.contains(pattern_str, regex=True)].copy()
        filtered_df.reset_index(names = 'filtered_index', inplace=True)
        
        filtered_df.loc[:,'onset'] = filtered_df.loc[:,'phonetic_wo_tone'].map(lambda x: re.sub(pattern_str, r'\1', x))
        filtered_df.loc[:,'endings'] = filtered_df.loc[:,'phonetic_wo_tone'].map(lambda x: re.sub(pattern_str, r'\2\3', x))
        _DATASET_INSIGHT_CACHE[cache_key] = filtered_df
        return filtered_df
    df.reset_index(names = 'filtered_index', inplace=True)
    _DATASET_INSIGHT_CACHE[cache_key] = df
    return df

def get_subclass_consonant_groups():
    mandarin_consonant_classes = ['r', 'sh', 'ch','s','z','j','zh','q','c','x']
    english_consonant_classes = ['ch', 'sh' ,'s', 't', 'z', 'dg', 'r']
    consonant_data_raw = [
        "r /ʐ/ /ɹ/",
        "sh /ʂ/ /ʃ/",
        "ch /tʂʰ/ /ʧ/",
        's /s/ /s/',
        'z /ts/ /s/',
        "j /tɕ/ /ʤ/",
        "zh /tʂ/ /ʧ/",
        'q /tɕʰ/ /ʧ/',
        "c /tsʰ/ /s/",
        "c /tsʰ/ /t/",
        "x /ɕ/ /ʃ/",
        "x /ɕ/ /z/"
    ]

    consonant_data = []
    for entry in consonant_data_raw:
        parts = entry.split()
        parts = [x.strip('/') for x in parts]
        consonant_data.append(parts)

    from itertools import groupby
    data = np.array(consonant_data)
    groups = []

    data_heading = {0: 'pinyin_ortho',
                    1:'mandarin', 
                    2:'english'}
    
    phoneme_lang_lookup = {1: 'english', 2:'mandarin'}

    for i in [1,2]:
        sorted_data = data[data[:, i].argsort()]
        # Group rows based on the third column
        grouped_data = {key: np.array(list(group)) for key, group in groupby(sorted_data, key=lambda x: x[i])}
        for key in grouped_data.keys():
            if len(grouped_data[key]) >1:
                groups.append({key: grouped_data[key],
                               phoneme_lang_lookup[i]: i})
    return groups, data_heading

def process_emb_filename(emb_file: str or os.PathLike,
                     mode = 'alldata',
                     seed = '42',
                     contrast = 'tone',
                     results_path = 'results'):
    modelname = os.path.basename(emb_file).split('_')[0] if 'checkpoint' not in emb_file else '-'.join(os.path.basename(emb_file).split('_')[:-2])
    datasetname = 'thchs30' if 'thchs30' in emb_file else 'vivos'
    flatten_flag = 'flat' if 'flat' in emb_file else 'original'
    cnn_flag = 'cnn' if 'cnn' in emb_file else ''
    seed_flag = f'seed-{seed}' if seed != 42 else ''
    segment_input_flag = 'segment-input' if 'segment-' in emb_file else ''
    outflag = "_".join(filter(None, (modelname, datasetname, flatten_flag, cnn_flag, segment_input_flag, seed_flag, mode,contrast, f"classification.pkl")))
    abs_save_path = os.path.join(results_path, f"{outflag}")

    return abs_save_path

def load_input_and_labels_and_mask(all_inputs_arr, all_labels_arr, rs, 
                                   contrast = 'tone', 
                                   mode = 'heldout', 
                                   group = None, 
                                   dataset_insight_path = './thchs30_transformed_dataset.csv'):

    contrast_dict = {'tone': {'column_filter':'phonetic_wo_tone',
                               'filter_consonant': False,
                               'label': 'tone_label'},
                     'consonant':{'column_filter':'endings',
                               'filter_consonant': True,
                               'label': 'onset'},}
    if mode == 'alldata':
        mask = np.zeros(1)
        X, y = all_inputs_arr, all_labels_arr

    elif mode == 'heldout':
        
        raw_df = read_dataset_insight(dataset_insight_path = dataset_insight_path,
                                      filter_consonant=contrast_dict[contrast]['filter_consonant'])

        if group:
            df = raw_df[raw_df[contrast_dict[contrast]['label']].isin(group)].copy()
        else:
            df = raw_df.copy()
        filtered_indices = df['filtered_index'].to_numpy()
        filtered_labels = df[contrast_dict[contrast]['label']].to_numpy().astype(str) if contrast == 'consonant' else all_labels_arr[filtered_indices]
        # filtered_labels = np.array([int(x) if x.isdigit() else 0 for x in filtered_labels]) if contrast == 'tone' else filtered_labels
        X, y = all_inputs_arr[filtered_indices], filtered_labels

        all_entries = df[contrast_dict[contrast]['column_filter']].to_numpy().astype(str)
        unique_entries, counts = np.unique(all_entries, return_counts=True)
        # the following block randomly samples 20% of all unique no-tone-pinyins
        # and constructs mask to make train test split
        test_percent = 0.2
        num_test = round(test_percent * len(unique_entries))
        test_entries = rs.choice(unique_entries, size = num_test)
        mask = np.isin(all_entries, test_entries)

    return X, y, mask

def run_classification(emb_file, 
                       mode, 
                       tgt_layers,
                       seed,
                       contrast,
                       results_path,
                       rewrite = False
                       ):
    if not os.path.isdir(results_path): 
        os.mkdir(results_path)
    rs = RandomState(MT19937(SeedSequence(seed))) # setting the random state for the random choice generator

    abs_save_path = process_emb_filename(emb_file, mode = mode,seed = seed, contrast=contrast, results_path=results_path)
    if os.path.isfile(abs_save_path) and not rewrite:
        tqdm.write(f'{abs_save_path} results already exist!') 
        return
    elif not os.path.isfile(emb_file):
        return
    else:
        tqdm.write(f'running classification, saving results to {abs_save_path}!')
        _, _, all_inputs_arr, all_labels_arr = torch.load(emb_file)
        if 'thchs30' in emb_file:
            dataset_insight_path = './thchs30_transformed_dataset.csv'
        elif 'vivos' in emb_file:
            dataset_insight_path = './vivos-train_transformed_dataset.csv'
        X, y, mask_array = load_input_and_labels_and_mask(all_inputs_arr, all_labels_arr,rs, 
                                                          contrast = contrast, 
                                                          mode = mode, 
                                                          dataset_insight_path=dataset_insight_path)
        results = classification_pipeline(X, y, seed=seed, mask_array=mask_array, tgt_layers=tgt_layers)
        with open(abs_save_path, 'wb') as file:
            pickle.dump(results, file)
        # torch.save(results, abs_save_path)
        tqdm.write(f'finished classification on {emb_file} and saved results to {abs_save_path}!')


def run_subclass(emb_file = './data/facebook-wav2vec2-base_thchs30_extracted-data.pt',
                        mode = 'heldout',
                        seed = 42,
                        tgt_layers=None, 
                        contrast = "tone",
                        results_path = 'results/subclass_experiment'):

    abs_save_path = process_emb_filename(emb_file, mode = mode,seed = seed, contrast=contrast, results_path=results_path)
    if not os.path.isdir(results_path): 
        os.mkdir(results_path)
    if os.path.isfile(abs_save_path):
        tqdm.write(f'{abs_save_path} results already exist!') 
        return
    elif not os.path.isfile(emb_file):
        return
    else:
        tqdm.write(f'running classification, saving results to {abs_save_path}!')
        _, _, all_inputs_arr, all_labels_arr = torch.load(emb_file)
        rs = RandomState(MT19937(SeedSequence(seed)))

        experiment_results = []
        from itertools import combinations
        if contrast == 'tone':
            groups = list(combinations(['1', '2', '3', '4'], r =2))
        elif contrast == 'consonant':
            raw_groups, data_heading = get_subclass_consonant_groups()
            groups = []
            consonant_classes = []
            for group in raw_groups:
                phoneme, language = group.keys()
                if language == 'english':
                    continue
                orthography = group[phoneme][:,0]
                classes = group[phoneme][:,list(data_heading.values()).index(language)]
                groups.append(tuple(orthography))
                consonant_classes.append(classes)

        for group in tqdm(groups,desc='Groupings'):
            tqdm.write(f"doing {contrast} subclass classification on {'-'.join(group)}")
            X, y, mask = load_input_and_labels_and_mask(all_inputs_arr, 
                                                        all_labels_arr, 
                                                        rs, 
                                                        contrast = contrast, 
                                                        mode = 'heldout', group = group)

            assert len(X) == len(y)

            results = classification_pipeline(X, y, seed=seed, mask_array=mask, tgt_layers=tgt_layers)
            experiment_results.extend([x|{'group': '-'.join(group)} for x in results])

        with open(abs_save_path, 'wb') as file:
            pickle.dump(experiment_results, file)
        tqdm.write(f'finished classification on {emb_file} and saved results to {abs_save_path}!')

def main():
    seed = 42
    mode = 'heldout'
    tgt_layers = None
    data_path = 'data/'
    results_path = f'results/{mode}_experiment'
    
    all_emb_files = [x for x in glob.glob(data_path + "*.pt") if 'extracted-data' in x                  
                 and 'flat' not in x
                 and 'cnn' not in x
                 and 'checkpoint' not in x
                 and 'segment-input' not in x]
    baselines_files = [x for x in all_emb_files if ('f0' in x or 'mfcc' in x) and 'concat' not in x]
    emb_files = [x for x in all_emb_files if 'f0' not in x and not 'mfcc' in x]
    str_emb_files = "\n".join(emb_files)
    print(f"Running classification on: \n{str_emb_files}")
    try:
        get_ipython
        emb_files.reverse()
    except:
        emb_files = emb_files

    for emb_file in tqdm(emb_files, desc='Embedding files'):
        for contrast in ['consonant', 'tone']:
            if ('vivos') in emb_file and (contrast == 'consonant'):
                continue
            run_classification(emb_file, mode, tgt_layers,seed, 
                               contrast=contrast, results_path = results_path, rewrite = False)

    
    baseline_results_path = 'results/baselines_results'
    for emb_file in tqdm(baselines_files, desc='Baseline files'):
        for contrast in ['consonant', 'tone']:
            run_classification(emb_file, mode, tgt_layers,seed,
            contrast=contrast, results_path = baseline_results_path, rewrite = False)
            
    


    # contrasts = ['tone'] #, 'consonant']
    # for contrast in contrasts:
    #     subclass_results_path = f'./results/heldout_experiment_{contrast}_subclass'
    #     for emb_file in tqdm(emb_files, desc='Embedding files'):
    #         run_subclass(emb_file=emb_file, mode = mode, seed = seed, tgt_layers=tgt_layers, contrast = contrast, results_path=subclass_results_path)





if __name__ == "__main__":
    main()
 