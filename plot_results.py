import glob
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as p9
from plotnine import (aes, arrow, element_blank, facet_wrap, geom_hline,
                      geom_line, geom_point, geom_smooth, ggplot, ggtitle,
                      guide_legend, guides, labs, scale_linetype_manual,
                      scale_x_continuous, theme, xlim, ylim, scale_color_discrete,
                      scale_y_continuous, scale_color_manual)
from sklearn.metrics import ConfusionMatrixDisplay

model_rename_dict = {'TencentGameMate-chinese-wav2vec2-base': 'Chinese-TGM', 
                'bert-base-chinese': 'BERT',
                'facebook-wav2vec2-base': 'English', 
                'kehanlu-mandarin-wav2vec2': 'Mandarin',
                'nguyenvulebinh-wav2vec2-base-vi':'Vietnamese',
                'patrickvonplaten-wav2vec2-base-random': 'Random-Init',
                'wcfr-wav2vec2-conformer-rel-pos-base-cantonese': 'Cantonese',
                'LeBenchmark-wav2vec2-FR-1K-base': 'French',
                'LeBenchmark-wav2vec2-FR-2.6K-base': 'French-2.6K',
                'LeBenchmark-wav2vec2-FR-3K-base': 'French-3K',
                'LeBenchmark-wav2vec2-FR-7K-base': 'French-7K',
                }

def parse_filename(filename):

    split_filename = filename.split('_')
    modelname = split_filename[0]
    datasetname = split_filename[1]
    mode = 'alldata' if 'alldata' in filename else 'heldout'
    flatten_flag = 'flat' if 'flat' in filename else 'original'
    cnn_flag = 'cnn' if 'cnn' in filename else None
    contrast = 'tone' if 'tone' in split_filename else 'consonant'
    seed_flag = 'seed' if 'seed' in filename else 42
    segment_flag = 'segment-input' if 'input' in filename else 'segment-output'
    fine_tuning_flag = ['960h', 'aishell1', 'vlsp2020']
    training_obj = 'fine-tuned' if any(x in modelname for x in fine_tuning_flag) else 'pre-trained'
    model = '-'.join(modelname.split('-')[:-1]) if training_obj == 'fine-tuned' else modelname

    flags = [modelname, datasetname, flatten_flag, cnn_flag,seed_flag, segment_flag, contrast, training_obj,model, mode]
    flag_keys = ["modelname", "datasetname", "flatten_flag", "cnn_flag","seed_flag", "segment_flag", 'contrast', 'training_obj', 'model', 'mode']

    return dict(zip(flag_keys, flags))


def read_results(results_path = 'results/heldout_experiment'):
    
    result_files = glob.glob(results_path + "/*.pkl")
    all_results = []
    for result_file in result_files:
        with open(result_file, 'rb') as file:
            loaded_results = pickle.load(file)
        result_basename = os.path.basename(result_file)
        flags = parse_filename(result_basename)
        if type(loaded_results[0]) == list: 
            loaded_results = list(np.array(loaded_results).flatten())
        all_results.extend([flags|x for x in loaded_results])
        if 'random' in flags['modelname']:
            flags['training_obj'] = 'fine-tuned'
            all_results.extend([flags|x for x in loaded_results])

    return all_results

def find_color_palette():
    all_results = read_results()
    df = pd.DataFrame(all_results).drop(['cm','cnn_flag'],axis = 1)
    df.model = df.model.map(lambda x: model_rename_dict[x])
    df = df[~(df.model.str.contains('\d+K'))]
    
    color_palette = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                    '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']

    modelnames = list(df.model.unique())
    modelnames.sort()
    # Map the colors to "modelname" values
    color_mapping = dict(zip(modelnames, color_palette))

    color_mapping['mfcc'] = '#db57d3'
    color_mapping['f0'] = '#5f57db'
    color_mapping['MFCC'] = '#db57d3'
    color_mapping['F0'] = '#5f57db'
    color_mapping['mfcc-ryant'] = '#db57d3'
    color_mapping['f0-ryant'] = '#5f57db'

    return color_mapping

def plot_probe_perf_plot(plot_df, 
                    selector, 
                    facet = None, 
                    linetype = None,
                    show_baseline = True,
                    not_show_legend = False,
                    color_mapping = None):
    
    x_min, x_max = plot_df.layer.min(), plot_df.layer.max()
    
    plot = (ggplot(plot_df, aes(x='layer', y='acc_score', color = 'model'))
        + geom_point() 
        + geom_line()
        + guides(color=guide_legend(ncol=4, title_position="left"))
        + labs(x='Transformer Layer', y='Accuracy')
        + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
        + theme(dpi=300,         
                legend_title=element_blank(),
                # legend_direction='horizontal',
                legend_position='bottom',
                # legend_box_spacing=0.25,
                figure_size=(5, 4)
                )
        )
    
    if facet: 
        plot += facet_wrap(facet, labeller=lambda label: label.capitalize())
    if linetype:
        plot += aes(linetype = linetype)
        plot += scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
        plot += guides(linetype = guide_legend(nrow = 2, byrow = True))
    if show_baseline:
        baselines_df = get_baseline_df()
        plot += geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
    if not_show_legend:
        plot += theme(legend_position='none')
    if color_mapping:
        plot += scale_color_manual(values=color_mapping)

    return plot

def sanity_check_plot():
    check_results = read_results('results/sanity_check')
    df = pd.DataFrame(check_results)
    df['cm_array'] = df['cm'].map(lambda x: x['array'])
    df['cm_labels'] = df['cm'].map(lambda x: x['labels'])
    drop_cols = ['datasetname', 'flatten_flag', 'cnn_flag', 'seed_flag', 'segment_flag', 'layer', 'training_obj', 'modelname','cm']
    viewing_df = df.drop(columns= drop_cols)

def plot_experiment1():
    all_results = read_results()
    df = pd.DataFrame(all_results).drop(['cm','cnn_flag'],axis = 1)
    df.model = df.model.map(lambda x: model_rename_dict[x])
    df = df[~(df.model.str.contains('\d+K'))]
    color_mapping = find_color_palette()
    
    datasetname = 'thchs30'
    
    # Plotting pretraining vs ft
    selector = 'tone'
    plot_df = df[(df.isin({selector}).any(axis=1)) & 
                 (df.segment_flag.str.contains('output')) &
                 (df.model.str.contains('Mandarin|English|BERT')) & 
                 (df.datasetname.str.contains(datasetname))].copy()
    facet = ['training_obj']
    plot_df["training_obj"] = pd.Categorical(plot_df['training_obj'])
    plot_df['training_obj'] = plot_df['training_obj'].cat.reorder_categories(['pre-trained','fine-tuned'])
    plot_df = plot_df.sort_values(by='training_obj')
    plot = plot_probe_perf_plot(plot_df, selector, 
                           show_baseline=True,
                           facet = facet,
                        #    linetype = 'training_obj',
                           color_mapping=color_mapping)
    print(plot)
    plot.save(f'results/probing_results_PTvsFT_{selector}.png')


    #Plotting tonal languages
    selector = 'tone'
    plot_df = df[(df.isin({selector}).any(axis=1)) & 
                 (df.segment_flag.str.contains('output')) &
                 (df.model.str.contains('Viet|Cantonese|English|French'))&
                 (df.training_obj.str.contains('pre'))&
                 (df.datasetname.str.contains(datasetname))]
    
    tonal_lang_lookup = {'Mandarin': 'Tonal',
                         'Cantonese': 'Tonal',
                         'Vietnamese': "Tonal",
                         'English':'Non-tonal',
                         'French': 'Non-tonal',
                         'BERT': 'Non-tonal',
                         'Random-Init': 'Non-tonal'}

    plot_df['tonality'] = plot_df.model.map(lambda x: tonal_lang_lookup[x.split('-')[0]])
    facet = ['tonality']
    plot = plot_probe_perf_plot(plot_df, selector, 
                           facet=facet,
                           show_baseline=True,
                        #    linetype = 'training_obj',
                           color_mapping=color_mapping)
    print(plot)
    plot.save(f'results/probing_results_TonevsNoTone_{selector}.png')

    # Plotting context contrasts
    selector = 'tone'
    plot_df = df[(df.isin({selector}).any(axis=1)) & 
                 (df.training_obj.str.contains('pre-trained')) &
                 (df.model.str.contains('Mandarin|English|BERT')) & 
                 (df.datasetname.str.contains(datasetname))]
    facet = ['segment_flag']

    plot = plot_probe_perf_plot(plot_df, selector, 
                           facet = facet, 
                           show_baseline=True,
                           color_mapping=color_mapping)
    print(plot)
    plot.save(f'results/probing_results_context_{selector}.png')

def plot_probe_perf():
    all_results = read_results()
    df = pd.DataFrame(all_results).drop(['cm','datasetname', 'cnn_flag'],axis = 1)
    baselines_df = get_baseline_df()

    x_min, x_max = df.layer.min(), df.layer.max()
    for selector in ['consonant', 'tone', 'segment-input', 'segment-output']:
        plot = (ggplot(df[df.isin({selector}).any(axis=1)], aes(x='layer', y='acc_score', color = 'model', linetype = 'training_obj'))
            + geom_point() 
            + geom_line()
            + facet_wrap(['contrast', 'segment_flag'])
            + guides(color=guide_legend(nrow=3,byrow=True, title_position="left"),
                    linetype = guide_legend(nrow = 2, byrow = True))
            + geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
            + labs(x='Transformer Layer', y='Accuracy')
            + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
            + scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
            + theme(dpi=300,         
                    legend_title=element_blank(),
                    legend_position='bottom',
                    # legend_box_spacing=0.25,
                    )
            + ggtitle(f'Probing experiment {selector}')
            )
        print(plot)
        plot.save(f'results/probing_results_{selector}.png')

def get_baseline_df(baseline_path = None,
                    only_ryant = True):
    if baseline_path is None:
        baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'results', 'baselines_results')
    baseline_path = os.path.expanduser(baseline_path)
    baselines = read_results(baseline_path)
    baselines_df = pd.DataFrame(baselines)
    copy_baseline = baselines_df.copy()
    copy_baseline['segment_flag']='segment-input'
    baselines_df = pd.concat((baselines_df,copy_baseline)).sort_values(by=['model'])
    copy_baseline = baselines_df.copy()
    copy_baseline['training_obj']='fine-tuned'
    baselines_df = pd.concat((baselines_df,copy_baseline)).sort_values(by=['model'])
    baselines_df["training_obj"] = pd.Categorical(baselines_df['training_obj'])
    baselines_df['training_obj'] = baselines_df['training_obj'].cat.reorder_categories(['pre-trained','fine-tuned'])
    baselines_df['model'] = baselines_df['model'].map(lambda x: x.split('-')[0].upper())
    if only_ryant:
        baselines_df = baselines_df[~baselines_df.apply(lambda row: row.astype(str).str.contains('resampled|mean')).any(axis=1)].copy()
    return baselines_df

def find_pretrain_color_palette():
    results_path = './results/pretrained_pipeline_results'
    all_results = read_results(results_path)
    df = pd.DataFrame(all_results)
    df['training_data'] = df.model.map(lambda x: x.split('-')[0])

    
    color_palette = ['#7fc97f', '#fdc086']

    datasetname = list(df.training_data.unique())
    # Map the colors to "modelname" values
    color_mapping = dict(zip(datasetname, color_palette))

    color_mapping['mfcc'] = '#db57d3'
    color_mapping['f0'] = '#5f57db'

    return color_mapping

def get_pretrain_conversion(file_pattern = '*epoch-to-updates.txt'):
    epoch_to_update_conversion_files = glob.glob(file_pattern)
    conversion_dicts = {}
    for conv_txt in epoch_to_update_conversion_files:
        with open(conv_txt, 'r') as f:
            content = f.read().splitlines()
        content_unzipped = list(zip(*[x.split()[-5:] for x in content]))
        epoch_to_update_conversion = dict(zip(content_unzipped[1], content_unzipped[3]))
        conversion_dicts[conv_txt.split('-')[0]] = epoch_to_update_conversion
    
    english_dict = {**conversion_dicts['ls960h'], **conversion_dicts['librispeech']}
    mandarin_dict = {**conversion_dicts['mandarin'], **conversion_dicts['magicdata']}
    return {'librispeech': english_dict,
            'magicdata': mandarin_dict}
    return conversion_dicts

def plot_pretrain():

    results_path = './results/pretrained_pipeline_results'
    all_results = read_results(results_path)
    df = pd.DataFrame(all_results).drop(['cm','datasetname', 'cnn_flag'],axis = 1)
    df['epoch'] = df.model.map(lambda x: x.split('-')[1].replace('checkpoint',''))
    df['training_data'] = df.model.map(lambda x: x.split('-')[0])
    conversion_dicts = get_pretrain_conversion()
    
    df['num_steps'] = df.apply(lambda x: conversion_dicts[x['training_data']][x['epoch']], axis = 1).astype('int')
    df['epoch'] = df['epoch'].astype('int')
    df = df.sort_values(by=['num_steps','layer','modelname',]).reset_index(drop = True)
    baselines_df = get_baseline_df()
    
    plot_df = df.loc[df.groupby(['modelname','num_steps', 'contrast'])['acc_score'].idxmax()].copy()

    metric = 'acc_score'
    color_mapping = find_pretrain_color_palette()
    pretrainplot = (ggplot(plot_df, aes(x='num_steps', y=metric, color = 'training_data'))
                    + geom_point() 
                    + geom_line()
                    + facet_wrap(['contrast'])
                    + geom_hline(baselines_df, aes(yintercept='acc_score', color='model'))
                    + labs(x='Num. training steps', y=metric)
                    + theme(dpi=300,         
                            legend_title=element_blank(),
                            # legend_direction='horizontal',
                            legend_position='bottom',
                            # legend_box_spacing=0.25,
                            )
                    + ggtitle('Model pretraining')
                    + scale_color_manual(values=color_mapping)

                    )
    print(pretrainplot)
    pretrainplot.save('./results/pretraining_consonant_vs_tone.png')

def plot_confusion_matrices():
    all_results = read_results()
    fig_save_path = 'results/confusion_matrices/'
    if not os.path.isdir(fig_save_path):
        os.mkdir(fig_save_path)


    # for result in all_results:
    #     if result['layer'] == 6:
    #         cm_data = result['cm']
    #         cm_normalized = cm_data['array'] / cm_data['array'].sum(axis=1)[:, np.newaxis]
    #         cmdisplay = ConfusionMatrixDisplay(np.around(cm_normalized, decimals=2), 
    #                                            display_labels=cm_data['labels'],)
    #         cmdisplay.plot()
    #         cmdisplay.ax_.set_title(f"{result['modelname']} Confusion Matrix for {result['contrast']} ")
    #         cmdisplay.figure_.savefig(os.path.join(fig_save_path,f"{result['modelname']}-{result['contrast']}.png"), dpi=300)
    #         plt.show()

    df_all_results = pd.DataFrame(all_results)
    df_all_results['cm_array'] = df_all_results['cm'].map(lambda x: x['array'])
    df_all_results['cm_labels'] = df_all_results['cm'].map(lambda x: x['labels'])
    df_all_results = df_all_results.sort_values(by = ['segment_flag','model', 'layer', 'training_obj'])
    for contrast in ['consonant', 'tone']:
        # for segment_flag in ['segment-input', 'segment-output']:
        for segment_flag in ['segment-output']:
            data = df_all_results[(df_all_results.isin({contrast}).any(axis=1) )&
                                  (df_all_results.isin({segment_flag}).any(axis=1) )&
                                (df_all_results.layer == 6)].sort_values(by='modelname').reset_index(drop = True)

            n_panels = len(data)
            row, column = 2, int(math.ceil(n_panels/2))
            fig, axs = plt.subplots(row, column, figsize=(5 * column, 4.5 * row))
            for i in range(n_panels):
                column_i, row_i   = int(i/row), int(i/column)
                row_i = i%2
                entry = data.iloc[i]
                cm_normalized = entry['cm_array'] / entry['cm_array'].sum(axis=1)[:, np.newaxis]
                df_cm = pd.DataFrame(np.around(cm_normalized, decimals=2), index=entry['cm_labels'],
                                    columns=entry['cm_labels'])
                cbar = True if column_i == (column - 1) else False
                sns.heatmap(df_cm, annot=True, ax=axs[row_i, column_i], vmin=0, vmax=1, cbar=cbar)
                axs[row_i, column_i].set_title(entry['modelname'])
            fig.suptitle(f'Confusion Matrix for {contrast}-{segment_flag}')
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(fig_save_path,f"heatmap-{contrast}-{segment_flag}.png"), dpi=300)

def read_alldata_outfile(outfile):
    df = pd.read_csv(outfile, index_col=0)
    modelname = os.path.basename(outfile).split('_')[0]
    fine_tuning_flag = ['960h', 'aishell1', 'vlsp2020']
    for x in fine_tuning_flag:
        if x in modelname:
            modelname = modelname.replace(x, 'ft')
            break
    df['segmentation'] = 'segment-input' if 'segment-input' in outfile else 'segment-output'
    df['modelname'] = modelname
    df = df.drop(['cm','mse','model_alpha'], axis = 1)

    df['training'] = 'fine-tuned' if 'ft' in modelname else 'pre-trained'
    df['model'] = modelname.replace('-ft', '') if 'ft' in modelname else modelname
    df['model'] = df.model.map(lambda x: x.replace('TencentGameMate', 'TGM'))
    df['model'] = df.model.map(lambda x: x.replace('conformer-rel-pos-', ''))    
    df['model'] = df.model.map(lambda x: '-'.join(x.split('-')[-3:]))    

    return df

def plot_layerwise_pretrain():
    filepath = 'results/pretrain_results'
    glob_pattern = f"{filepath}/*.out"
    out_files = [x for x in glob.glob(glob_pattern) if 'tone-class' in x and 'cnn' not in x]
    out_files.sort()
    
    joined_df = pd.DataFrame()
    for outfile in out_files:
        df = read_alldata_outfile(outfile)
        df['training_data'] = df['modelname'].map(lambda x: x.split('-')[0])
        df['epoch'] = df['modelname'].map(lambda x: x.split('-')[1].replace('checkpoint',''))
        joined_df = pd.concat((joined_df, df))
    joined_df = joined_df.reset_index(drop = True)


    conversion_dicts = get_pretrain_conversion()
    joined_df['num_steps'] = joined_df.apply(lambda x: conversion_dicts[x['training_data']][x['epoch']], axis = 1).astype('int')
    joined_df['epoch'] = joined_df['epoch'].astype('int')
    joined_df.sort_values(by=['num_steps','layer','modelname',]).reset_index()
    
    
    baseline_path = './results/baselines_results'
    baselines = read_results(baseline_path)
    baselines_df = pd.DataFrame(baselines)

    maj_error = get_maj_error(seed = 42, contrast='tone')
    joined_df['err'] = (maj_error + joined_df['acc_score'] - 1)/maj_error

    metric_dict = {#'err': "Error reduction rate",
                   'acc_score': "Accuracy"}

    for metric in metric_dict.keys():
        plot_df = joined_df
        plot_df["epoch"] = pd.Categorical(plot_df['epoch'])
        x_min, x_max = plot_df.layer.min(), plot_df.layer.max()
        plot = (ggplot(plot_df, aes(x='layer', y=metric, color = 'epoch'))
            + geom_point(show_legend=False) 
            + geom_line(show_legend=False)
            + facet_wrap(['segmentation', 'training_data'])
            + geom_hline(baselines_df[baselines_df.contrast == 'tone'], aes(yintercept='acc_score', linetype='model'))
            + labs(x='Transformer Layer', y=metric_dict[metric])
            + theme(dpi=300,         
                        legend_title=element_blank(),
                        legend_position='bottom',
                        # legend_box_spacing=0.25,
                        )
            + guides(color=guide_legend(reverse=True))
            + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
            + ggtitle(f"{metric_dict[metric]} plot with pre-training")
            )  
        print(plot)
        plot.save(f"results/pretraining_accuracy_datasetcomparison.png")

def get_maj_error(seed = 42, contrast = 'tone'):
    from numpy.random import MT19937, RandomState, SeedSequence
    from classification import read_dataset_insight

    rs = RandomState(MT19937(SeedSequence(seed)))
    filter_consonant = True if contrast == 'consonant' else False
    df = read_dataset_insight(filter_consonant=filter_consonant)

    all_pinyins = df.pinyin.map(lambda x: x[:-1]).to_numpy(dtype=str)
    labels = df.pinyin.map(lambda x: x[-1]).to_numpy(dtype=str)
    pinyin, counts = np.unique(all_pinyins, return_counts=True)
    test_percent = 0.2
    test_samples = round(test_percent * len(pinyin))
    test_pinyins = rs.choice(pinyin, size = test_samples)
    mask_array = np.isin(all_pinyins, test_pinyins)
    _, counts = np.unique(labels[mask_array], return_counts=1)
    maj_base = max(counts)/sum(counts)
    maj_error = 1-maj_base

    return maj_error

def plot_subclass(contrast = 'consonant'):
    subclass_result_path = f'./results/heldout_experiment_{contrast}_subclass'
    subclass_results = read_results(subclass_result_path)
    df = pd.DataFrame(subclass_results).drop(['model_alpha','datasetname', 'cnn_flag', 'flatten_flag'],axis = 1)
    df = df.sort_values(by = ['group', 'segment_flag','model', 'layer', 'training_obj'])
    df = df[~(df.model == 'LeBenchmark-wav2vec2-FR-7K-base')]

    baselines_df = df[df.modelname.str.contains('f0|mfcc')].reset_index(drop=True)
    df = df[~df.modelname.str.contains('f0|mfcc')].reset_index(drop=True)
    df.model = df.model.map(lambda x: model_rename_dict[x])
    color_mapping = find_color_palette()

    
    x_min, x_max = df.layer.min(), df.layer.max()

    for selector in ['segment-output']:
        plot_df = df[(df.isin({selector}).any(axis=1)) & 
                     (df.model.str.contains('Mandarin|English')) & 
                     (~df.training_obj.str.contains('fine-tuned'))].copy()
        group_order = plot_df.groupby('group')['acc_score'].std().sort_values(ascending=False).index
        plot_df['group'] = pd.Categorical(plot_df['group'], categories=group_order, ordered=True)
        plot = (ggplot(plot_df, aes(x='layer', y='acc_score', color = 'model'))
            + geom_point() 
            + geom_line()
            + facet_wrap('~group', ncol=len(group_order)) 
            + guides(color=guide_legend(nrow=1,byrow=True, title_position="left"),
                    linetype = guide_legend(nrow = 2, byrow = True))
            # + geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
            + labs(x='Transformer Layer', y='Accuracy')
            + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
            + scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
            + theme(dpi=300,         
                    legend_title=element_blank(),
                    # legend_direction='horizontal',
                    legend_position='bottom',
                    figure_size=(10, 4)
                    # legend_box_spacing=0.25,
                    )
            # + ggtitle(f'{contrast.capitalize()} subclass experiment {selector}')
            + scale_color_manual(values=color_mapping)
            )
        print(plot)
        plot.save(f'results/{contrast}_subclass_{selector}.png')

        # heatmap_df = plot_df[plot_df.layer == 6].copy()
        heatmap_df = plot_df.loc[plot_df.groupby(['modelname', 'group'])['acc_score'].idxmax()].copy()
        heatmap_df = heatmap_df[~heatmap_df.modelname.str.contains('Tencent')]
        heatmap_df = heatmap_df.sort_values(by = ['group', 'segment_flag','model', 'layer', 'training_obj'])
        heatmap_df['cm_array'] = heatmap_df['cm'].map(lambda x: x['array'])
        heatmap_df['cm_labels'] = heatmap_df['cm'].map(lambda x: x['labels'])

        n_panels = len(heatmap_df)
        row = len(heatmap_df.group.unique())
        column = int(n_panels/row)
        fig, axs = plt.subplots(row, column, figsize=(5 * column, 4.5 * row))
        column_i = 0
        for i in range(n_panels):
            entry = heatmap_df.iloc[i]
            cm_normalized = entry['cm_array'] / entry['cm_array'].sum(axis=1)[:, np.newaxis]
            df_cm = pd.DataFrame(np.around(cm_normalized, decimals=2), index=entry['cm_labels'],
                                columns=entry['cm_labels'])
            
            row_i =  int(i/column)
            column_i =0 if column_i == column else column_i
            cbar = True if column_i == (column-1) else False
            sns.heatmap(df_cm, annot=True, ax=axs[row_i, column_i], vmin=0, vmax=1, cbar=cbar)
            axs[row_i, column_i].set_title(f"{entry['modelname']} Layer {entry['layer']}")
            column_i+=1
        fig.suptitle(f'Confusion Matrix for {contrast} subclass experiment')
        plt.tight_layout()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        fig.savefig(os.path.join('results/confusion_matrices',f"heatmap-{contrast}-subclass-{selector}-best_layer.png"), dpi=300)

def plot_subclass_pretrain(contrast = 'consonant'):

    subclass_result_path = f'./results/pretrained_pipeline_results_{contrast}_subclass'
    subclass_results = read_results(subclass_result_path)
    df = pd.DataFrame(subclass_results).drop(['model_alpha','datasetname', 'cnn_flag', 'flatten_flag'],axis = 1)
    df = df.sort_values(by = ['group', 'segment_flag','model', 'layer', 'training_obj'])

    baselines_df = df[df.modelname.str.contains('f0|mfcc')].reset_index(drop=True)
    df = df[~df.modelname.str.contains('f0|mfcc')].reset_index(drop=True)

    df['epoch'] = df.model.map(lambda x: x.split('-')[1].replace('checkpoint',''))
    df['training_data'] = df.model.map(lambda x: x.split('-')[0])
    conversion_dicts = get_pretrain_conversion()
    df['num_steps'] = df.apply(lambda x: conversion_dicts[x['training_data']][x['epoch']], axis = 1).astype('int')
    df['epoch'] = df['epoch'].astype('int')
    df = df.sort_values(by=['num_steps','layer','modelname',]).reset_index(drop = True)
    baselines_df = get_baseline_df()

    plot_df = df[(df.contrast == contrast) & (df.segment_flag.str.contains('output'))]
    plot_df = plot_df.loc[plot_df.groupby(['modelname', 'group'])['acc_score'].idxmax()].copy()
    acc_plot = (ggplot(plot_df, aes(x='num_steps', y='acc_score', color = 'training_data'))
                    + geom_point() 
                    + geom_line()
                    + facet_wrap(['group'])
                    # + geom_hline(baselines_df, aes(yintercept='acc_score', color='model'))
                    + labs(x='Num. training steps', y='acc_score')
                    + theme(dpi=300,         
                            legend_title=element_blank(),
                            # legend_direction='horizontal',
                            legend_position='bottom',
                            # legend_box_spacing=0.25,
                            )
                    + ggtitle('Model pretraining')
                    )
    print(acc_plot)
    acc_plot.save(f'results/probing_results_pretraining_{contrast}_subclass.png')

    heatmap_df = plot_df.copy()
    heatmap_df = heatmap_df[heatmap_df.segment_flag.str.contains('output')]
    heatmap_df['cm_array'] = heatmap_df['cm'].map(lambda x: x['array'])
    heatmap_df['cm_labels'] = heatmap_df['cm'].map(lambda x: x['labels'])
    heatmap_df = heatmap_df.sort_values(by=['group','model','epoch'])
    heatmap_df = heatmap_df.drop(labels = ['seed_flag','segment_flag', 'contrast', 'training_obj', 'modelname','model'], axis = 1)
    def get_total_recall(cm):
        false_positives = np.zeros(len(cm))
        false_negatives = np.zeros(len(cm))
        for i in range(len(cm)):
            # Sum of true positives for the current class
            tp = cm[i, i]
            # Sum of false positives for the current class
            false_positives[i] = np.sum(cm[:, i]) - tp
            # Sum of false negatives for the current class
            false_negatives[i] = np.sum(cm[i, :]) - tp

            recall = list(cm.diagonal()/cm.sum(axis=0))
        # print(false_negatives, false_positives)
        all_false_values = np.sum(false_negatives) + np.sum(false_positives)
        normalized_false_negatives = np.sum(false_negatives) /np.sum(cm)
        return recall

    heatmap_df['normalized_recall'] = heatmap_df.cm_array.map(lambda x: get_total_recall(x))
    heatmap_df = heatmap_df.explode(['normalized_recall','cm_labels']).reset_index(drop=True)
    heatmap_df = pd.melt(heatmap_df, id_vars=['training_data','epoch','num_steps','group', 'cm_labels', 'layer'], value_vars=['normalized_recall']).reset_index(drop = True)
    heatmap_df['value'] = heatmap_df['value'].astype(float)
    heatmap_df.head()
    # heatmap_df
    (ggplot(heatmap_df, aes(x = 'num_steps', y = 'value', color = 'training_data', shape='cm_labels')) 
        + geom_point()
        + geom_line()
        + facet_wrap(['group']))


    heatmap_df = plot_df.copy()
    heatmap_df['cm_array'] = heatmap_df['cm'].map(lambda x: x['array'])
    heatmap_df['cm_labels'] = heatmap_df['cm'].map(lambda x: x['labels'])
    heatmap_df = heatmap_df.sort_values(by=['group','model','epoch'])
    def get_total_recall(cm):
        false_positives = np.zeros(len(cm))
        false_negatives = np.zeros(len(cm))
        for i in range(len(cm)):
            # Sum of true positives for the current class
            tp = cm[i, i]
            # Sum of false positives for the current class
            false_positives[i] = np.sum(cm[:, i]) - tp
            # Sum of false negatives for the current class
            false_negatives[i] = np.sum(cm[i, :]) - tp

            recall = list(cm.diagonal()/cm.sum(axis=0))
        # print(false_negatives, false_positives)
        all_false_values = np.sum(false_negatives) + np.sum(false_positives)
        normalized_false_negatives = np.sum(false_negatives) /np.sum(cm)
        return recall
    
    heatmap_df['normalized_recall'] = heatmap_df.cm_array.map(lambda x: get_total_recall(x))
    # heatmap_df.plot(x = 'epoch', y = 'normalized_recall', aes = aes(color = 'Group'))
    false_prediction_plot = (ggplot(heatmap_df, aes(x = 'num_steps', y = 'normalized_recall', color = 'training_data')) 
        + geom_point()
        + geom_line()
        + facet_wrap(['group']))
    
    print(false_prediction_plot)
    false_prediction_plot.save(f'results/false_prediction_pretraining_{contrast}_subclass.png')
    
    # n_panels = len(heatmap_df)
    # row = len(heatmap_df.group.unique())
    # column = int(n_panels/row)
    # fig, axs = plt.subplots(row, column, figsize=(5 * column, 4.5 * row))
    # column_i = 0
    # for i in range(n_panels):
    #     entry = heatmap_df.iloc[i]
    #     cm_normalized = entry['cm_array'] / entry['cm_array'].sum(axis=1)[:, np.newaxis]
    #     df_cm = pd.DataFrame(np.around(cm_normalized, decimals=2), index=entry['cm_labels'],
    #                         columns=entry['cm_labels'])
        
    #     row_i =  int(i/column)
    #     column_i =0 if column_i == 5 else column_i
    #     cbar = True if column_i == 4 else False
    #     sns.heatmap(df_cm, annot=True, ax=axs[row_i, column_i], vmin=0, vmax=1, cbar=cbar)
    #     axs[row_i, column_i].set_title(entry['modelname'])
    #     column_i+=1
    # fig.suptitle(f'Confusion Matrix for consonant subclass experiment in pretraining models')
    # fig.set_size_inches()
    # plt.tight_layout()
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()
    # fig.savefig(os.path.join('results/confusion_matrices',f"heatmap-consonant-subclass-pretrain.png"), dpi=300)



def main():
    plot_experiment1()
    pass

if __name__ == "__main__":
    main()