import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (confusion_matrix, f1_score)
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
            raise ValueError(f"mask array len: {len(mask_array)} != X len {len(X)}\n if this is not the correct behavior please check configurations")
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
        list: All results in a list of dictionaries
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
        print(f"train:{len(y_train)}, test: {len(y_test)}   ")
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

def read_dataset_insight(dataset_insight_path = './thchs30_transformed_dataset.csv', 
                         filter_consonant = False):
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
        return filtered_df
    df.reset_index(names = 'filtered_index', inplace=True)
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

def process_emb_filename(emb_file: str | os.PathLike,
                     mode = 'alldata',
                     seed = '42',
                     contrast = 'tone',
                     results_path = 'results',
                     subclass = None):
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

def process_raw_input_dataset(raw_input_dataset, rs, 
                              contrast = 'tone', 
                              mode = 'heldout',
                              group = None, 
                              ):

    contrast_dict = {'tone': {'column_filter':'phonetic_wo_tone',
                            'filter_consonant': False,
                            'label': 'tone_labels'},
                    'consonant':{'column_filter':'ending',
                            'filter_consonant': True,
                            'label': 'onset'},}
    inputs, labels, column_filter = zip(*[(x['embs'], 
                        x[contrast_dict[contrast]['label']], 
                        x[contrast_dict[contrast]['column_filter']]) 
                            for x in tqdm(raw_input_dataset, desc='loading raw dataset')])
    
    inputs = np.concatenate(inputs, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    column_filter = np.concatenate(column_filter, axis = 0)



    if contrast == 'consonant':
        consonants = ['r', 'sh', 'ch','s','z','j','zh','q','c','x']
        consonant_filter_indices = np.isin(labels, consonants)
        inputs, labels, column_filter = inputs[consonant_filter_indices], labels[consonant_filter_indices], column_filter[consonant_filter_indices]

    if mode == 'alldata':
        mask = np.zeros(1)
    elif mode == 'heldout':
        if group:
            group = [int(x) if str(x).isdigit() else x for x in group ]
            group_filter = np.isin(labels, group)
            inputs, labels, column_filter = inputs[group_filter], labels[group_filter], column_filter[group_filter]
        # the following block randomly samples 20% of all unique values in column_filter
        # and constructs mask to make train test split
        unique_entries, counts = np.unique(column_filter, return_counts=True)
        test_percent = 0.2
        num_test = round(test_percent * len(unique_entries))
        test_entries = rs.choice(unique_entries, size = num_test)
        mask = np.isin(column_filter, test_entries)

    while len(inputs.shape) < 3:
        # Accounting for the shapes of audio features (f0, mfccs)
        inputs = np.expand_dims(inputs, axis = 1)
    return inputs, labels, mask


def run_classification(emb_file = './classifier_input/facebook-wav2vec2-base_thchs30_extracted-data.pt', 
                       mode = 'heldout', 
                       tgt_layers = None,
                       seed = 42,
                       contrast = 'tone',
                       results_path = 'results/heldout_experiment',
                       rewrite = False,
                       subclass = False
                       ):
    if not os.path.isdir(results_path): 
        os.mkdir(results_path)
    rs = RandomState(MT19937(SeedSequence(seed))) # setting the random state for the random choice generator

    abs_save_path = process_emb_filename(emb_file, mode = mode,seed = seed, contrast=contrast, results_path=results_path)
    if os.path.isfile(abs_save_path) and not rewrite:
        tqdm.write(f'{abs_save_path} results already exist!') 
        return
    elif not os.path.isfile(emb_file):
        raise FileNotFoundError(f"{emb_file} not found")
    else:
        tqdm.write(f'running classification, saving results to {abs_save_path}!')
        raw_input_dataset = torch.load(emb_file)
        X, y, mask_array = process_raw_input_dataset(raw_input_dataset, rs, 
                                                    contrast = contrast, 
                                                    mode = mode,)
        results = classification_pipeline(X, y, seed=seed, mask_array=mask_array, tgt_layers=tgt_layers)
        with open(abs_save_path, 'wb') as file:
            pickle.dump(results, file)
        tqdm.write(f'finished classification on {emb_file} and saved results to {abs_save_path}!')

def get_subclass_groups(contrast):
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
    return groups
    

def run_subclass(emb_file = './classifier_input/facebook-wav2vec2-base_thchs30_extracted-data.pt',
                mode = 'heldout',
                seed = 42,
                tgt_layers=None, 
                contrast = "tone",
                results_path = 'results/heldout_experiment_tone_subclass'):

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
        rs = RandomState(MT19937(SeedSequence(seed)))
        raw_input_dataset = torch.load(emb_file)
        experiment_results = []
        groups = get_subclass_groups(contrast)
        for group in tqdm(groups,desc='Groupings'):
            tqdm.write(f"doing {contrast} subclass classification on {'-'.join(group)}")
            X, y, mask_array = process_raw_input_dataset(raw_input_dataset, rs, 
                                            contrast = contrast, 
                                            mode = mode,group = group)

            assert len(X) == len(y)

            results = classification_pipeline(X, y, seed=seed, mask_array=mask_array, tgt_layers=tgt_layers)
            experiment_results.extend([x|{'group': '-'.join(group)} for x in results])
        with open(abs_save_path, 'wb') as file:
            pickle.dump(experiment_results, file)
        tqdm.write(f'finished classification on {emb_file} and saved results to {abs_save_path}!')


def get_classifier_input_stats():
    for dataset in ['thchs30', 'vivos-train']:
        emb_file = f'classifier_input/facebook-wav2vec2-base_{dataset}_extracted-data.pt'
        seed = 42
        mode = 'heldout'
        rs = RandomState(MT19937(SeedSequence(seed)))
        raw_input_dataset = torch.load(emb_file)
        for contrast in ['tone', 'consonant']:
            groups = get_subclass_groups(contrast) if 'thchs30' in emb_file else []
            groups.append(None)
            for group in groups:
                layer = 0
                # raw_input_dataset = torch.load(emb_file)
                X, y, mask_array = process_raw_input_dataset(raw_input_dataset, rs, 
                                            contrast = contrast, 
                                            mode = mode,group = group)
                X_ = np.nan_to_num(X[:,layer,:])
                X_train, X_test, y_train, y_test = custom_train_test_split(X_, 
                                                                            y, 
                                                                            mask_array = mask_array, 
                                                                            seed = seed)
                lookup = {'train':y_train,
                        'test':y_test}
                for split in lookup.keys():
                    labels, counts = np.unique(lookup[split], return_counts=True)
                    total_count = len(lookup[split])
                    counts_normalized = counts/total_count*100
                    counts_percent = np.around(counts_normalized, decimals=2, out=None)
                    print(f"""
the {split} split for {contrast} classification in {emb_file} has the following stats:
Total number of samples: {total_count}
Class distribution: {dict(zip(labels,counts_percent))}
""")

def main():
    seed = 42
    mode = 'heldout'
    tgt_layers = None
    data_path = 'classifier_input/'
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
            if ('vivos' in emb_file) and (contrast == 'consonant'):
                continue
            # for baseline_mode in ['alldata', 'heldout']:
            baseline_mode = 'heldout'
            run_classification(emb_file, baseline_mode, tgt_layers,seed,
                contrast=contrast, results_path = baseline_results_path, rewrite = True)
            
    contrasts = ['tone', 'consonant']
    for contrast in contrasts:
        subclass_results_path = f'./results/heldout_experiment_{contrast}_subclass'
        for emb_file in tqdm(emb_files+baselines_files, desc='Embedding files'):
            if 'vivos' in emb_file:
                continue
            run_subclass(emb_file=emb_file, mode = mode, seed = seed, tgt_layers=tgt_layers, contrast = contrast, results_path=subclass_results_path)


def train_test_stats():
    stats_files = [ 'classifier_input/f0_thchs30_extracted-data.pt',
                   'classifier_input/f0_vivos-train_extracted-data.pt']
    seed = 42
    mode = 'heldout'
    tgt_layers = None
    for emb_file in stats_files:
        for contrast in ['tone', 'consonant']:
            if ('vivos' in emb_file) and ('consonant' in contrast):
                continue
            rs = RandomState(MT19937(SeedSequence(seed)))
            raw_input_dataset = torch.load(emb_file)
            experiment_results = []
            if 'thchs' in emb_file:
                groups = get_subclass_groups(contrast)
                groups.append(None)
            else:
                groups = [None]
            for group in groups:
                # tqdm.write(f"doing {contrast} subclass classification on {'-'.join(group)}")
                X, y, mask_array = process_raw_input_dataset(raw_input_dataset, rs, 
                                                contrast = contrast, 
                                                mode = mode,group = group)

                assert len(X) == len(y)

                results = classification_pipeline(X, y, seed=seed, mask_array=mask_array, tgt_layers=tgt_layers)
                group = 'none' if not group else group
                print(f"group:{group}, emb_file: {emb_file}")


if __name__ == "__main__":
    main()
 