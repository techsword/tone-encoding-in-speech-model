import argparse
import gc
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from embgen import run_embgen
# from classification import run_classification, run_subclass

from generate_classifier_input import run_embgen
from experiment_classification import run_classification, run_subclass


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
        "--results_path",
        type=str,
        default='results/',
        help="where to save the results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='heldout',
        help="options: heldout or alldata",
    )
    parser.add_argument(
        "--contrast",
        type=str,
        default='tone',
        help="options: heldout or alldata",
    )
    parser.add_argument(
        "--tgt_layers",
        type=int,
        default=None,
        help="options: heldout or alldata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="options: heldout or alldata",
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
        help="If extract from CNN layers.",
    )
    parser.add_argument(
        "--subclass",
        action = 'store_true',
        help="If run subclass experiment.",
    )
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    emb_filename = run_embgen(model_ID = args.model_name,
                              datasetname=args.dataset_name,
                              flattened=args.flattened, 
                              cnn = args.cnn, 
                              segment_input = args.segment_input)
    if not os.path.isdir(args.results_path):
        os.mkdir(args.results_path)
    class_results_path = os.path.join(args.results_path, f'{args.mode}_experiment')
    run_classification(emb_filename, 
                       mode = args.mode, 
                       tgt_layers=args.tgt_layers, 
                       seed=args.seed, 
                       results_path=class_results_path,
                       contrast = args.contrast)
    if args.subclass:
        subclass_results_path = class_results_path + f'_{args.contrast}_subclass'
        run_subclass(emb_filename, 
                        mode = args.mode, 
                        tgt_layers=args.tgt_layers, 
                        seed=args.seed, 
                        results_path=subclass_results_path,
                        contrast = args.contrast)


    
if __name__ == "__main__":
    main()