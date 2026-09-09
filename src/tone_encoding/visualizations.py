import glob
import math
import os
import pickle
# import patchworklib as pw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as p9
from plotnine import (aes, arrow, element_blank, facet_wrap, geom_hline,
                      geom_line, geom_point, geom_smooth, geom_tile, ggplot, ggtitle,
                      guide_legend, guides, labs, scale_linetype_manual,
                      scale_x_continuous, theme, xlim, ylim, scale_color_discrete,
                      scale_y_continuous, scale_color_manual)
from sklearn.metrics import ConfusionMatrixDisplay
from tone_encoding.plot_results import parse_pretrain_modelname


model_rename_dict = {'TencentGameMate-chinese-wav2vec2-base': 'Chinese-TGM', 
                'bert-base-chinese': 'BERT-Chinese',
                'facebook-wav2vec2-base': 'English', 
                'kehanlu-mandarin-wav2vec2': 'Mandarin',
                'nguyenvulebinh-wav2vec2-base-vi':'Vietnamese',
                'patrickvonplaten-wav2vec2-base-random': 'Random-Init',
                'wcfr-wav2vec2-conformer-rel-pos-base-cantonese': 'Cantonese',
                'LeBenchmark-wav2vec2-FR-1K-base': 'French',
                'LeBenchmark-wav2vec2-FR-2.6K-base': 'French-2.6K',
                'LeBenchmark-wav2vec2-FR-3K-base': 'French-3K',
                'LeBenchmark-wav2vec2-FR-7K-base': 'French-7K',
                'mfcc':'MFCC',
                'f0':'F0'
                }

tonal_lang_lookup = {'Mandarin': 'Tonal',
                        'Chinese': 'Tonal',
                    'Cantonese': 'Tonal',
                    'Vietnamese': "Tonal",
                    'English':'Non-tonal',
                    'French': 'Non-tonal',
                    'BERT': 'Tonal',
                    'Random-Init': 'Non-tonal'}

color_mapping = {'BERT': '#a6cee3',
 'Cantonese': '#1f78b4',
 'Chinese-TGM': '#b2df8a',
 'English': '#33a02c',
 'French': '#fb9a99',
 'Mandarin': '#e31a1c',
 'Vietnamese': '#fdbf6f',
 'MFCC': '#db57d3',
 'F0': '#5f57db'}
movetolast = lambda l, e: [x for x in l if x != e] + [e]
def subclass_label_func(label):
    if label[0].isdigit():
        return 'T'+label
    else:
        return label

def read_results(results_path = 'results/'):
    result_files = glob.glob(results_path + "/**/*.pkl", recursive=True)
    all_results = []
    for result_file in result_files:
        with open(result_file, 'rb') as file:
            loaded_results = pickle.load(file)
        filename = os.path.basename(result_file)
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
        model = model_rename_dict.get(model) 

        flags_values = [modelname, datasetname, flatten_flag, cnn_flag,seed_flag, segment_flag, contrast, training_obj,model, mode]
        flag_keys = ["modelname", "datasetname", "flatten_flag", "cnn_flag","seed_flag", "segment_flag", 'contrast', 'training_obj', 'model', 'mode']
        flags = dict(zip(flag_keys, flags_values))
        if type(loaded_results[0]) == list: 
            loaded_results = list(np.array(loaded_results).flatten())
        all_results.extend([flags|x for x in loaded_results])
        if 'random' in flags['modelname']:
            flags['training_obj'] = 'fine-tuned'
            all_results.extend([flags|x for x in loaded_results])
    return all_results

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
                # legend_background=p9.element_rect(size=2, fill='none'),
                legend_box_spacing=0.01,
                legend_text=p9.element_text(size=8),
                figure_size=(5, 3.5)
                )
        )
    
    if facet: 
        plot += facet_wrap(facet, labeller=lambda label: label.capitalize())
    if linetype:
        plot += aes(linetype = linetype)
        plot += scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
        plot += guides(linetype = guide_legend(nrow = 2, byrow = True))

    if not_show_legend:
        plot += theme(legend_position='none')
    if color_mapping:
        plot += scale_color_manual(values=color_mapping)

    return plot

def get_baseline_df(baseline_path = None,
                    no_extra = True,
                    datasetname = None):
    if baseline_path is None:
        baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'results', 'baselines_results')
    baseline_path = os.path.expanduser(baseline_path)
    baselines = read_results(baseline_path)
    baselines = [x for x in baselines if ('f0' in x['modelname']) or ('mfcc' in x['modelname'])]
    baselines_df = pd.DataFrame(baselines)
    facet_variable_lookup = {'segment_flag': 'segment-input',
                             'training_obj': 'fine-tuned'}
    for facet_variable_column in facet_variable_lookup.keys():
        copy_baseline = baselines_df.copy()
        copy_baseline[facet_variable_column]= facet_variable_lookup[facet_variable_column]
        baselines_df = pd.concat((baselines_df,copy_baseline))
    baselines_df["training_obj"] = pd.Categorical(baselines_df['training_obj'])
    baselines_df['training_obj'] = baselines_df['training_obj'].cat.reorder_categories(['pre-trained','fine-tuned'])
    baselines_df['model'] = baselines_df['model'].map(lambda x: x.split('-')[0].upper() if x else x)
    if no_extra:
        baselines_df = baselines_df[~baselines_df.apply(lambda row: row.astype(str).str.contains('resampled|mean')).any(axis=1)].copy()
    if datasetname:
        return baselines_df[baselines_df.datasetname.str.contains(datasetname)]
    return baselines_df

def plot_experiment1():
    all_results = read_results(results_path='results/heldout_experiment')
    df = pd.DataFrame(all_results).drop(['cm','cnn_flag'],axis = 1)
    # df.model = df.model.map(lambda x: model_rename_dict[x])
    df = df[~df.model.isna()]
    df = df[(~df.model.str.contains('\d+K')) & (~df.model.str.contains('MFCC|F0'))]
    # df = df[df.group.isna()]
    df =  df[df['mode'] == 'heldout']

    
    color_palette = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                    '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']

    modelnames = list(df.model.sort_values().unique())
    # Map the colors to "modelname" values
    color_mapping = dict(zip(modelnames, color_palette))
    color_mapping['MFCC'] = '#db57d3'
    color_mapping['F0'] = '#5f57db'

    df['tonality'] = df.model.map(lambda x: tonal_lang_lookup[x.split('-')[0]])
    
    for datasetname in ['vivos', 'thchs30']:
        language = 'Vietnamese' if datasetname == 'vivos' else 'Mandarin'
        baselines_df = get_baseline_df(datasetname=datasetname)
        baselines_df =  baselines_df[baselines_df['mode'] == 'heldout']


        #Plotting tonal vs non-tonal languages
        selector = 'tone'
        plot_df = df[(df.isin({selector}).any(axis=1)) & 
                    (df.segment_flag.str.contains('output')) &
                    (df.model.str.contains('Viet|Mandarin|Cantonese|English|French|BERT'))&
                    (df.training_obj.str.contains('pre'))&
                    (df.datasetname.str.contains(datasetname))].reset_index()
        model_order = ['F0', 'MFCC'] + [x[0] for x in list(plot_df.groupby(by=['tonality', 'model']).model.unique())] 
        if 'BERT' in model_order:
            model_order = movetolast(model_order, 'BERT')
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
        baselines_df['model'] = pd.Categorical(baselines_df['model'], categories=model_order, ordered=True)
        
        facet = ['tonality']
        plot = plot_probe_perf_plot(plot_df, selector, 
                            facet=facet,
                            show_baseline=True,
                            #    linetype = 'training_obj',
                            color_mapping=color_mapping)
        plot += geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
        plot += theme(figure_size=(5, 3.6))
        plot += ylim(0.3,1)
        print(plot)
        selector = '_'.join((selector,datasetname)) if datasetname =='vivos' else selector
        plot.save(f'results/probing_results_TonevsNoTone_{selector}.png')


        # Plotting pretraining vs ft
        selector = 'tone'
        plot_df = df[(df.isin({selector}).any(axis=1)) & 
                    (df.segment_flag.str.contains('output')) &
                    (df.model.str.contains(f'{language}|English')) & 
                    (df.datasetname.str.contains(datasetname))].copy()
        facet = ['training_obj']
        plot_df["training_obj"] = pd.Categorical(plot_df['training_obj'])
        plot_df['training_obj'] = plot_df['training_obj'].cat.reorder_categories(['pre-trained','fine-tuned'])
        model_order = ['F0', 'MFCC'] + [x[0] for x in list(plot_df.groupby(by=['tonality', 'model']).model.unique())] 
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
        baselines_df['model'] = pd.Categorical(baselines_df['model'], categories=model_order, ordered=True)
        
        # plot_df = plot_df.sort_values(by='training_obj')
        plot = plot_probe_perf_plot(plot_df, selector, 
                            show_baseline=True,
                            facet = facet,
                            #    linetype = 'training_obj',
                            color_mapping=color_mapping)
        
        plot += geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
        plot += ylim(0.3,1)
        print(plot)
        selector = '_'.join((selector,datasetname)) if datasetname =='vivos' else selector
        plot.save(f'results/probing_results_PTvsFT_{selector}.png')

        # Plotting context contrasts
        if language == 'Mandarin':
            for selector in ['tone', 'consonant']:
                plot_df = df[(df.isin({selector}).any(axis=1)) & 
                            (df.training_obj.str.contains('pre-trained')) &
                            (df.model.str.contains('Mandarin|English|BERT')) & 
                            (df.datasetname.str.contains(datasetname))]
                facet = ['segment_flag']

                plot = plot_probe_perf_plot(plot_df, selector, 
                                    facet = facet, 
                                    show_baseline=True,
                                    color_mapping=color_mapping)
                plot += geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
                plot += ggtitle(f'{selector.capitalize()} segmentation test')
                print(plot)
                selector = '_'.join((selector,datasetname)) if datasetname =='vivos' else selector
                plot.save(f'results/probing_results_context_{selector}.png')
    return color_mapping

def plot_pretrain(color_mapping):
    results_path = './results/pretrained_pipeline_results/heldout_experiment'
    all_results = read_results(results_path)
    drop_cols = ['cm','datasetname', 'cnn_flag', 'flatten_flag', 'seed_flag','segment_flag','training_obj','model','mode']
    df = pd.DataFrame(all_results).drop(drop_cols,axis = 1)
    parsed = df.modelname.map(parse_pretrain_modelname)
    df['training_data'] = [x[0] for x in parsed]
    df['epoch'] = [x[1] for x in parsed]
    df['num_steps'] = [x[2] for x in parsed]
    df['num_steps'] = df['num_steps'].astype(int)
    df = df[df['num_steps']%10000 == 5000] # only using *5000 checkpoints

    df = df.sort_values(by=['training_data','num_steps','layer','modelname',]).reset_index(drop = True)
    training_data_to_lang = {'magicdata': 'Mandarin',
                             'librispeech': 'English'}
    df['training_data'] = df['training_data'].map(lambda x: training_data_to_lang[x.lower()])

    baselines_df = get_baseline_df(datasetname='thchs30')
    baselines_df = baselines_df[baselines_df['mode'] == 'heldout']
    model_order = ['F0', 'MFCC','English', 'Mandarin'] 
    df['training_data'] = pd.Categorical(df['training_data'], categories=model_order, ordered=True)
    baselines_df['model'] = pd.Categorical(baselines_df['model'], categories=model_order, ordered=True)
    metric = 'acc_score'
    plot_df = df.loc[df.groupby(['modelname','num_steps', 'contrast'])[metric].idxmax()].copy()

    pretrainplot = (ggplot(plot_df, aes(x='num_steps', y=metric, color = 'training_data'))
                    + geom_point() 
                    + geom_line()
                    + facet_wrap(['contrast'])
                    + geom_hline(baselines_df, aes(yintercept=metric, color='model'))
                    # + geom_hline(baselines_df, aes(yintercept=0.9))
                    + labs(x='Num. training steps', y= 'Accuracy')
                    + theme(dpi=300,         
                            legend_title=element_blank(),
                            legend_position='bottom',
                            legend_box_spacing=0.01,
                        legend_text=p9.element_text(size=8),
                            figure_size=(5, 3.5)
                            )
                    # + ggtitle('Model pretraining')
                    + scale_color_manual(values=color_mapping)
                    )
    print(pretrainplot)
    pretrainplot.save('./results/pretraining_consonant_vs_tone.png')

def plot_confusion_matrices():
    all_results = read_results(results_path='results/heldout_experiment')
    fig_save_path = 'results/confusion_matrices/'
    if not os.path.isdir(fig_save_path):
        os.mkdir(fig_save_path)

    df = pd.DataFrame(all_results)
    df['cm_array'], df['cm_labels'] = zip(*df['cm'].map(lambda x: (x['array'], x['labels'])))
    # df['model'] = df['model'].map(lambda x: model_rename_dict[x])
    df = df.loc[df.training_obj == 'pre-trained'].reset_index(drop=True)
    df = df.iloc[(df.groupby(['modelname', 'contrast', 'segment_flag'])['acc_score'].idxmax())].copy().reset_index(drop=True)
    for contrast in ['consonant', 'tone']:
        segment_flag = 'segment-output'
        datasetname = 'thchs30'
        plot_df = df[(df.datasetname == datasetname) &
                  (df.segment_flag == segment_flag) &
                  (df.contrast == contrast) ].copy().sort_values(by = ['model']).reset_index(drop=True)
        
                  
        n_panels = len(plot_df)
        row, column = 2, int(math.ceil(n_panels/2))
        fig, axs = plt.subplots(row, column, figsize=(5 * column, 4.5 * row))
        for i in range(n_panels):
            column_i, row_i   = int(i/row), int(i/column)
            row_i = i%2
            entry = plot_df.iloc[i]
            cm_normalized = entry['cm_array'] / entry['cm_array'].sum(axis=1)[:, np.newaxis]
            df_cm = pd.DataFrame(np.around(cm_normalized, decimals=2), index=entry['cm_labels'],
                                columns=entry['cm_labels'])
            cbar = True if column_i == (column - 1) else False
            sns.heatmap(df_cm, annot=True, ax=axs[row_i, column_i], vmin=0, vmax=1, cbar=cbar)
            axs[row_i, column_i].set_title(entry['model'])
        fig.suptitle(f'Confusion Matrix for {contrast}-{segment_flag}')
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(fig_save_path,f"heatmap-{contrast}-{segment_flag}.png"), dpi=300)

def plot_subclass(color_mapping):
    for contrast in ['tone', 'consonant']:
        results_path = f'./results/heldout_experiment_{contrast}_subclass'
        all_results = read_results(results_path=results_path)
        df = pd.DataFrame(all_results).drop(['cm','cnn_flag'],axis = 1)
        df.model = df.model.map(lambda x: model_rename_dict.get(x) if model_rename_dict.get(x) else x)

        df = df[~df.model.isna()]
        baselines_df = df[df.model.str.contains('F0|MFCC')].copy()
        df = df[(~df.model.str.contains('\d+K')) & (df.model.str.contains('Mandarin|English'))]
    

        x_min, x_max = df.layer.min(), df.layer.max()
        for selector in ['segment-output']:
            plot_df = df[(df.isin({selector}).any(axis=1)) & 
                        
                        (df.datasetname.str.contains('thchs')) & 
                        (~df.training_obj.str.contains('fine-tuned'))].copy()
            # group_order =
            # plot_df.groupby('group')['acc_score'].std().sort_values(ascending=False).index
            grouped_data = plot_df.groupby(['model','group'])['acc_score'].max()
            group_order = (grouped_data['Mandarin'] - grouped_data['English']).sort_values(ascending=False).index
            plot_df['group'] = pd.Categorical(plot_df['group'], categories=group_order, ordered=True)
            baselines_df['group'] = pd.Categorical(baselines_df['group'], categories=group_order, ordered=True)
            model_order = ['F0', 'MFCC'] + [x[0] for x in list(plot_df.groupby(by=['model']).model.unique())] 
            plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
            baselines_df['model'] = pd.Categorical(baselines_df['model'], categories=model_order, ordered=True)
            x_size = 5*len(group_order)/3
            if len(group_order) != 6:
                baselines_df = baselines_df[baselines_df.model.str.contains('MFCC')]
            plot = (ggplot(plot_df, aes(x='layer', y='acc_score', color = 'model'))
                + geom_point() 
                + geom_line()
                + facet_wrap('~group', ncol=len(group_order), labeller = subclass_label_func) 
                + guides(color=guide_legend(nrow=1,byrow=True, title_position="left"),
                        linetype = guide_legend(nrow = 2, byrow = True))
                + geom_hline(baselines_df[baselines_df.isin({selector}).any(axis=1)], aes(yintercept='acc_score', color='model'))
                + labs(x='Transformer Layer', y='Accuracy')
                + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
                + scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
                + theme(dpi=300,         
                        legend_title=element_blank(),
                        # legend_direction='horizontal',
                        legend_position='bottom',
                        figure_size=(x_size, 3.5),
                        legend_box_spacing=0.01,
                        legend_text=p9.element_text(size=8),
                        )
                # + ggtitle(f'{contrast.capitalize()} subclass experiment {selector}')
                + scale_color_manual(values=color_mapping)
                )
            print(plot)
            plot.save(f'results/{contrast}_subclass_{selector}.png')

def plot_pretrain_subclass(color_mapping):

    for contrast in ['consonant','tone']:
        results_path = f'./results/pretrained_pipeline_results/heldout_experiment_{contrast}_subclass'
        all_results = read_results(results_path=results_path)
        drop_cols = ['cm','datasetname', 'cnn_flag', 'flatten_flag', 'seed_flag','segment_flag','training_obj','model','mode']
        df = pd.DataFrame(all_results).drop(drop_cols,axis = 1)
        parsed = df.modelname.map(parse_pretrain_modelname)
        df['training_data'] = [x[0] for x in parsed]
        df['epoch'] = [x[1] for x in parsed]
        df['num_steps'] = [x[2] for x in parsed]
        df['num_steps'] = df['num_steps'].astype(int)
        df = df[df['num_steps']%10000 == 5000] # only using *5000 checkpoints

        df = df.sort_values(by=['training_data','num_steps','layer','modelname',]).reset_index(drop = True)
        training_data_to_lang = {'magicdata': 'Mandarin',
                                'librispeech': 'English'}
        df['training_data'] = df['training_data'].map(lambda x: training_data_to_lang[x.lower()])

        baselines_df = get_baseline_df(baseline_path= results_path.replace('pretrained_pipeline_results/',''),
                                       datasetname='thchs30')
        baselines_df = baselines_df[(baselines_df['mode'] == 'heldout') & (baselines_df['segment_flag'])]
    
        metric = 'acc_score'
        plot_df = df.loc[df.groupby(['training_data','num_steps', 'group', 'contrast'])[metric].idxmax()].copy()

        grouped_data = plot_df.groupby(['training_data','group'])['acc_score'].max()
        group_order = (grouped_data['Mandarin'] - grouped_data['English']).sort_values(ascending=False).index
        model_order = ['F0', 'MFCC', 'English', 'Mandarin'] 
        plot_df['training_data'] = pd.Categorical(plot_df['training_data'], categories=model_order, ordered=True)
        # group_order = plot_df.groupby('group')['acc_score'].std().sort_values(ascending=False).index
        plot_df['group'] = pd.Categorical(plot_df['group'], categories=group_order, ordered=True)
        baselines_df['group'] = pd.Categorical(baselines_df['group'], categories=group_order, ordered=True)
        x_size = 5*len(group_order)/3+0.5
        x_max = df.num_steps.max()

        if len(group_order) != 6:
            baselines_df = baselines_df[baselines_df.model.str.contains('MFCC')]
        plot = (ggplot(plot_df, aes(x='num_steps', y='acc_score', color = 'training_data'))
            + geom_point() 
            + geom_line()
            + facet_wrap('~group', ncol=len(group_order),labeller = subclass_label_func) 
            + guides(color=guide_legend(nrow=1,byrow=True, title_position="left"),
                    linetype = guide_legend(nrow = 2, byrow = True))
            + geom_hline(baselines_df, aes(yintercept='acc_score', color='model'))
            + labs(x='Num training steps', y='Accuracy')
            + scale_x_continuous(breaks=range(0, x_max+1, 20000), labels=[str(x)+'0k' for x in list(range(0, int(x_max/10000)+1,2))])
            + scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
            + theme(dpi=300,         
                    legend_title=element_blank(),
                    legend_position='bottom',
                    legend_box_spacing=0.01,
                    legend_text=p9.element_text(size=8),
                    figure_size=(x_size, 3.5),
                    )
            # + ggtitle(f'{contrast.capitalize()} subclass experiment {selector}')
            + scale_color_manual(values=color_mapping)
            )
        print(plot)
        plot.save(f'results/probing_results_pretraining_{contrast}_subclass.png')


def plot_alldata():
    for contrast in ['tone', 'consonant']:
        datasetname = 'thchs'
        results_path = f'./results/alldata_experiment'
        alldata_results = read_results(results_path=results_path)
        alldata_df = pd.DataFrame(alldata_results).drop(['cm','cnn_flag'],axis = 1)
        alldata_models = alldata_df.model.unique()
        heldout_results = read_results(results_path=results_path.replace('alldata','heldout'))
        heldout_df = pd.DataFrame(heldout_results).drop(['cm','cnn_flag'],axis = 1)
        heldout_df = heldout_df[heldout_df.model.isin(alldata_models)]

        df = pd.concat((alldata_df,heldout_df)).copy().reset_index()
        df = df[(df['contrast'] == contrast) &
                (df['datasetname'].str.contains(datasetname)) &
                (df.training_obj == 'pre-trained')]

        df.model = df.model.map(lambda x: model_rename_dict[x])
        df = df[~(df.model.str.contains('\d+K'))]
        baselines_df = get_baseline_df()
        baselines_df = baselines_df[(baselines_df['contrast'] == contrast) &
                                    (baselines_df['datasetname'].str.contains(datasetname))]

        plot_df = df.copy()
        facet = ['segment_flag', 'mode']
        # plot_df["training_obj"] = pd.Categorical(plot_df['training_obj'])
        # plot_df['training_obj'] = plot_df['training_obj'].cat.reorder_categories(['pre-trained','fine-tuned'])
        # plot_df = plot_df.sort_values(by='training_obj')
        plot = plot_probe_perf_plot(plot_df, datasetname, 
                            show_baseline=True,
                            facet = facet)
        
        plot += geom_hline(baselines_df, aes(yintercept='acc_score', color='model'))
        plot += ggtitle(f"""
{contrast.capitalize()} classification accuracy with {datasetname.upper()} 
Contextual information comparison 
""")
        print(plot)
        break

def plot_last_checkpoint():
    for contrast in ['consonant','tone']:
        results_path = f'./results/pretrained_pipeline_results/heldout_experiment_{contrast}_subclass'
        all_results = read_results(results_path=results_path)
        last_checkpoints = [x for x in all_results if '85000' in x['modelname']]
        drop_cols = ['cm','datasetname', 'cnn_flag', 'flatten_flag', 'seed_flag','segment_flag','training_obj','model','mode']
        df = pd.DataFrame(last_checkpoints).drop(drop_cols,axis = 1)

        parsed = df.modelname.map(parse_pretrain_modelname)
        df['model'] = [x[0] for x in parsed]
        df['epoch'] = [x[1] for x in parsed]
        df['num_steps'] = [x[2] for x in parsed]
        df['num_steps'] = df['num_steps'].astype(int)
        df = df[df['num_steps']%10000 == 5000] # only using *5000 checkpoints

        df = df.sort_values(by=['model','num_steps','layer','modelname',]).reset_index(drop = True)
        training_data_to_lang = {'magicdata': 'Mandarin',
                                'librispeech': 'English'}
        df['model'] = df['model'].map(lambda x: training_data_to_lang[x.lower()])

        baselines_df = get_baseline_df(baseline_path= results_path.replace('pretrained_pipeline_results/',''),
                                       datasetname='thchs30')
        baselines_df = baselines_df[(baselines_df['mode'] == 'heldout') & (baselines_df['segment_flag'])]
 
        metric = 'acc_score'
        x_min, x_max = df.layer.min(), df.layer.max()
        plot_df = df.copy()
        grouped_data = plot_df.groupby(['model','group'])[metric].max()
        group_order = (grouped_data['Mandarin'] - grouped_data['English']).sort_values(ascending=False).index
        plot_df['group'] = pd.Categorical(plot_df['group'], categories=group_order, ordered=True)
        baselines_df['group'] = pd.Categorical(baselines_df['group'], categories=group_order, ordered=True)

        max_df = plot_df.groupby(['model','group']).agg(max_score=(metric, 'max'))
        max_plot = sns.barplot(x="group", 
                y="max_score", 
                hue="model", 
                data=max_df) 
        max_plot.set(ylim=(0.8,1))

        x_size = 5*len(group_order)/3
        if len(group_order) != 6:
            baselines_df = baselines_df[baselines_df.model.str.contains('MFCC')]
        model_order =list(baselines_df.model.unique()) + ['English', 'Mandarin'] 
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
        baselines_df['model'] = pd.Categorical(baselines_df['model'], categories=model_order, ordered=True)
        plot = (ggplot(plot_df, aes(x='layer', y=metric, color = 'model'))
            + geom_point() 
            + geom_line()
            + facet_wrap('~group', ncol=len(group_order), labeller = subclass_label_func) 
            + guides(color=guide_legend(nrow=1,byrow=True, title_position="left"),
                    linetype = guide_legend(nrow = 2, byrow = True))
            + geom_hline(baselines_df, aes(yintercept=metric, color='model'))
            + labs(x='Transformer Layer', y='Accuracy')
            + scale_x_continuous(breaks=range(x_min, x_max+1,2), labels=range(x_min, x_max+1,2))
            + scale_linetype_manual(values={'pre-trained': 'solid', 'fine-tuned': 'dotted'})
            + theme(dpi=300,         
                    legend_title=element_blank(),
                    # legend_direction='horizontal',
                    legend_position='bottom',
                    figure_size=(x_size, 3.5),
                    legend_box_spacing=0.01,
                    legend_text=p9.element_text(size=8),
                    )
            # + ggtitle(f'{contrast.capitalize()} subclass experiment {selector}')
            + scale_color_manual(values=color_mapping)
            )
        print(plot)
        plot.save(f'results/{contrast}_subclass_checkpoint85000.png')

def plot_tone_contour():

    import parselmouth
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # sns.set() # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 300 # Show nicely large images in this notebook
    plt.figure(figsize=(5, 4))
    tones_audio_files = glob.glob('tone_recording/*.wav')
    tones_audio_files.sort()
    def draw_pitch(pitch):
            # Extract selected pitch contour, and
            # replace unvoiced samples by NaN to not plot
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values==0] = np.nan
            plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
            plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
            plt.grid(False)
            plt.ylim(0, pitch.ceiling)
            plt.ylabel("fundamental frequency [Hz]")

    tone_dict = []
    for i, tone_file in enumerate(tones_audio_files):
        snd = parselmouth.Sound(tone_file)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        time = pitch.xs()
        tone_label = os.path.basename(tone_file).split('.')[0].capitalize().replace('one','')
        plot = sns.lineplot(x=time, y = pitch_values, label = tone_label)
        plot.set(ylim=(50,200))

        tone_dict.append({'tone_label':tone_label,
        'time':time,'pitch_values': pitch_values})

    plt.xlabel('Time - Seconds')
    plt.ylabel('F0 values - Hertz')
    plt.show()

    plot.get_figure().savefig('results/tone_illustration.png')


    df = pd.DataFrame(tone_dict)
    return df


def main():
    # tone_recording_df =plot_tone_contour()
    color_mapping = plot_experiment1()
    plot_subclass(color_mapping)
    plot_pretrain(color_mapping)
    plot_pretrain_subclass(color_mapping)
    plot_confusion_matrices()
    pass

if __name__ == "__main__":
    main()