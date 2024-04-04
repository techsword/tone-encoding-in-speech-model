
import argparse
import gc
import glob
import math
import os
import re

import fairseq
import numpy as np
import pandas as pd
import textgrids
import torch
import torchaudio
from generate_aligned_dataset import save_aligned_dataset_csv
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_fairseq_model(checkpoint):
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
    original = model[0]
    imported = import_fairseq_model(original)
    return imported

def get_segmented_output(emb, segment_df, audio_len):
    shape = emb.shape
    total_frames = shape[-2]
    
    segment_df['startFrame'] = (segment_df['startTime']/audio_len*total_frames).map(math.ceil)
    segment_df['endFrame'] = (segment_df['endTime']/audio_len*total_frames).map(math.ceil)
    segment_df = segment_df[~segment_df['transcription'].str.contains('sil|SIL', na = False)]
    segment_dict = segment_df.iloc[:,3:].to_dict()

    segments = torch.zeros(shape[0],len(segment_df), shape[-1])
    for i, x in enumerate(segment_dict['transcription']):
        segment_start = segment_dict['startFrame'][x]
        segment_end = segment_dict['endFrame'][x]
        segment_tensor = torch.mean(emb[:,segment_start:segment_end,:],dim=1, keepdim=False)
        segments[:,i,:] = segment_tensor.clone()
    return segments

def get_segmented_input(wave, segment_df, audio_len):
    shape = wave.shape
    total_frames = shape[1]

    segment_df.loc[:,'startFrame'] = (segment_df['startTime']/audio_len*total_frames).map(math.ceil)
    segment_df.loc[:,'endFrame'] = (segment_df['endTime']/audio_len*total_frames).map(math.ceil)
    segment_df = segment_df[~segment_df['transcription'].str.contains('sil|SIL', na = False)]
    segment_dict = segment_df.iloc[:,3:].to_dict()

    def parse_transcription(x):
        segment_start = segment_dict['startFrame'][x]
        segment_end = segment_dict['endFrame'][x]
        segment_seq = wave[:,segment_start:segment_end].clone()
        return segment_seq

    segments = [parse_transcription(x) for i, x in enumerate(segment_dict['transcription'])]

    return segments

def get_hidden_cnn(input_values, model):
    cnn_feature_encoder = model.feature_extractor
    hidden_states = {}
    hidden_state = input_values[:, None]
    hidden_state.requires_grad = False
    layer = 0
    for conv_layer in cnn_feature_encoder.conv_layers:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_state = torch.utils.checkpoint.checkpoint(
            create_custom_forward(conv_layer),
            hidden_state,
        )
        hidden_states[layer] = hidden_state
        layer +=1

    # Define the number of frames in which the audio signal is split
    # courtesy of Charlotte Pouw
    num_frames = hidden_state.shape[-1]
    window_sizes = {
        0: 64,
        1: 32,
        2: 16,
        3: 8,
        4: 4,
        5: 2,
        6: 1
    }
    # Get moving averages of CNN layers
    averaged_cnn_layers = []
    for layer_idx in range(7):
        window_size = window_sizes[layer_idx]
        windows = torch.stack([hidden_states[layer_idx][0][:,window_size * y:window_size * (y + 1)] for y in range(num_frames)])
        windows = windows.movedim(-1,0)
        averaged_windows = torch.mean(windows, dim=0).cpu()
        averaged_cnn_layers.append(averaged_windows)
    return torch.stack(averaged_cnn_layers)

def generating_features(file_IDs, model, dataset_path, df, 
                        flattened = False, 
                        cnn = False, 
                        tokenizer = None, 
                        segment_input = False,
                        extension = '.wav'):
    audioname_list, feat_list = [], []
    model = model.to(device)
    model.eval()
    
    glob_list = [x for x in glob.glob(dataset_path + '/**/*' + extension, recursive = True)]
    if flattened:
        glob_list = [x for x in glob_list if 'flat' in x]
    else:
        glob_list = [x for x in glob_list if 'flat' not in x]
    for audioname in tqdm(glob_list):
        file_ID = os.path.basename(audioname).split('.')[0]
        if file_ID not in file_IDs:
            continue
        with torch.inference_mode():
            if tokenizer != None:
                with open(os.path.join(dataset_path, file_ID + ".wav.trn")) as file:
                    annot = file.read().splitlines()
                chars = re.sub(r'[^\u4e00-\u9FFF]+', '',annot[0]) # remove all non-chinese character from string
                if segment_input:
                    features = []
                    for char in chars:
                        inputs = tokenizer(char.capitalize(), return_tensors="pt").to(device)
                        outputs = model(**inputs, output_hidden_states = True)
                        feature = torch.stack(outputs.hidden_states).detach().cpu()[:,:,1:-1,:].squeeze()
                        features.append(feature)
                    features = torch.stack(features, dim = 1)
                else:
                    inputs = tokenizer(chars.capitalize(), return_tensors="pt").to(device)
                    outputs = model(**inputs, output_hidden_states = True)
                    features = torch.stack(outputs.hidden_states).detach().cpu()[:,:,1:-1,:].squeeze()
                features_batch = [features.numpy()]
            else:
                def get_audio_hidden_states(audio_input):
                    try:
                        if "transformers" in str(type(model)):
                            output = model(input_values = audio_input.to(device), output_hidden_states = True)
                            output_hidden_states = output.hidden_states
                        else:
                            output_hidden_states, _ = model.extract_features(audio_input.to(device))
                        return torch.stack(output_hidden_states).detach().cpu()
                    except:
                        output_hidden_states = torch.zeros(13,1,1,768) if "transformers" in str(type(model)) else torch.zeros(12,1,1,768)
                        print(f'{file_ID} has a weird segment')
                        return output_hidden_states

                segment_df = df[df.file_ID == file_ID].copy()
                wave, sr = torchaudio.load(audioname)
                audio_len = wave.shape[1]/sr
                
                if segment_input:
                    segments = get_segmented_input(wave, segment_df, audio_len)
                    output_hidden_states = [get_audio_hidden_states(audio_segment) for audio_segment in segments]
                    features = [torch.mean(x, dim = -2) for x in output_hidden_states]
                    features = torch.stack(features).movedim(2,1)
                    features_batch = [features[:,x,:,:].movedim(0, -2).numpy() for x in range(features.shape[1])]
                else:
                    if cnn:
                        cnn_hidden = get_hidden_cnn(wave.squeeze(1).to(device), model)
                        # cnn_hidden is a tensor with dimension (layers, num_frames, 512)
                        features = cnn_hidden.unsqueeze(1)
                    else:
                        features = get_audio_hidden_states(wave.squeeze(1))
                    features_batch = [get_segmented_output(features[:,x,:,:], segment_df, audio_len).numpy() for x in range(features.shape[1])]
                    assert len(segment_df[~segment_df['transcription'].str.contains('sil|SIL', na = False)]) == features_batch[0].shape[1]
                # features = torch.stack([get_segmented_output(x, segment_df, audio_len) for x in features]).numpy()
            feat_list.extend(features_batch)
            audioname_list.append(file_ID)
    print(f"{len(audioname_list)} audio files are used in the extracted dataset")
    return list(zip(audioname_list, feat_list))

class classifierInputDataset(Dataset):

    def __init__(self, features, datasetname = 'thchs30'):
        """_summary_

        Args:
            features (_type_): _description_
            datasetname (str, optional): 'thchs30' or 'vivos'. Defaults to 'thchs30'.
        """
        self.file_IDs, self.embs = zip(*features)
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
        seg_emb = np.moveaxis(self.embs[idx], 0 ,-2)
        trns = self.transformed_dataset[self.transformed_dataset[:,0] == file_ID]
        trn_chars = trns[:,3].astype(str)
        split_phonetic = trns[:,4].astype(str)
        phonetic_wo_tone = np.array([x[:-1] for x in split_phonetic])
        tone_labels = np.array([int(x[-1]) if x[-1].isdigit() else 0 for x in split_phonetic])

        onset = np.array(list(map(lambda x: re.sub(self.pattern_str, r'\1', x), phonetic_wo_tone)))
        ending = np.array(list(map(lambda x: re.sub(self.pattern_str, r'\2\3', x), phonetic_wo_tone)))

        return {'embs': seg_emb, 
                'chars': trn_chars, 
                'tone_labels': tone_labels,
                'onset': onset,
                'ending': ending,
                'split_phonetic' : split_phonetic, 
                'phonetic_wo_tone':phonetic_wo_tone, 
                'file_ID': file_ID}
    
def run_embgen(model_ID = 'facebook/wav2vec2-base', 
               datasetname = 'thchs30', 
               save_path = 'classifier_input', 
               flattened = False, 
               cnn = False, 
               segment_input = False):
    if 'thchs30' in datasetname:
        dataset_path = "~/data_thchs30/data/"
        dataset_path = os.path.expanduser(dataset_path)

    elif 'vivos' in datasetname:
        datasetname = 'vivos-train'
        dataset_path = "~/vivos/train/waves"
        dataset_path = os.path.expanduser(dataset_path)

    df = save_aligned_dataset_csv(dataset = datasetname,
                                rewrite=False)


    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    modelname = '-'.join(model_ID.split('/')[-2:]).replace('.pt','')
    flatten_flag = 'flat' if flattened else ''
    cnn_flag = 'cnn' if cnn else ''
    segment_input_flag = 'segment-input' if segment_input else ''
    save_name = "_".join(filter(None, (modelname, datasetname, flatten_flag, 'extracted-data', cnn_flag, segment_input_flag))) + '.pt'
    save_name = os.path.join(save_path, save_name)

    if os.path.isfile(save_name):
        print(f"{save_name} exists! skipping")
    else:
        print(f"generating classifier data input to {save_name}")
        file_IDs = df.file_ID.unique()
        if os.path.isfile(model_ID) and 'fairseq' in model_ID:
            model = load_fairseq_model(model_ID)
            tokenizer = None
        else:
            from transformers import AutoModel
            try:
                model = AutoModel.from_pretrained(model_ID)
                if "modeling_bert" in str(type(model)):
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_ID)
                else:
                    tokenizer = None                
            except:
                KeyError(f"{model_ID} is not part of the Huggingface Hub")
        features = generating_features(file_IDs, model, dataset_path=dataset_path, df=df, flattened=flattened, cnn=cnn, tokenizer=tokenizer, segment_input=segment_input)
        dataset = classifierInputDataset(features, datasetname = datasetname)
        metadata_dict = {'model_ID': model_ID, 
                         'datasetname': datasetname}
        dataset_with_metadata = list(map(lambda x: metadata_dict|x, dataset))
        torch.save(dataset_with_metadata, save_name, pickle_protocol = 4)
        print(f"finished!")
    
    return save_name

def parse_args():
    parser = argparse.ArgumentParser(description="Use a pre-trained wav2vec2 model to generate embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        default='facebook/wav2vec2-base',
        help="The name of the model to use (via the transformers library).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='thchs30',
        help="The name of the dataset to extract hiddenstate activations. ('thchs30' or 'vivos')"
    )
    parser.add_argument(
        "--flattened",
        action = 'store_true',
        help="If use original sound or sound with flattened pitch contour.",
    )
    parser.add_argument(
        "--cnn",
        action = 'store_true',
        help="If extract from CNN layers.",
    )
    parser.add_argument(
        "--segment_input",
        action = 'store_true',
        help="If use timestamp to slice audio before feeding audio as input to speech models.",
    )
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    run_embgen(model_ID = args.model_name,
                              datasetname=args.dataset_name,
                              flattened=args.flattened, 
                              cnn = args.cnn, 
                              segment_input = args.segment_input)

if __name__ == "__main__":
    main()
