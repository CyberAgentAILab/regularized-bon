import os
import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd


from utility_func import *
from utils import load_dataset, load_samples_from_file, matrix_dir #, approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix


if __name__ == "__main__":
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir
    
    n_lines = args.n_lines
    start_iter = args.start_iter
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # This defines the cost function used by the WD
    compute_similarity = load_similarity(sim)

    src_lines = load_dataset(dataset)
    trg_lines = load_dataset(dataset, ref=True)
    
    model_n = os.path.basename(model_name)


    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)


    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp, True, 4, 0.1)
    
    assert len(filtered_files) > 0

    print('first 10 files=', filtered_files[:10])

    rows = []

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break
        filename = filtered_files[sample_id]
        assert "{:04}".format(sample_id) in filename

        src_input = src_lines[sample_id]
        
        df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]
        df.fillna("", inplace=True)
        df['text'] = df['text'].astype(str)
        hyp = df.iloc[:]['text']
        
        matrix_filename = filename + "_" + sim + "_" + str(n_samples)
        matrix_path = os.path.join(matrix_dir, dataset, model_n, matrix_filename)
        matrix = compute_similarity.compute_score_matrix(hyp, src_input)
        np.savetxt(matrix_path, matrix)
        