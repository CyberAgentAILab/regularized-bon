import os
import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

from utility_func import *
from utils import load_dataset, load_matrix, load_samples_from_file, result_dir, matrix_dir, reward_dir
from parser import get_mbr_parser
from reward_model import load_reward_model


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
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

    reward_model_id = args.reward_model
    recompute_reward = args.recompute_reward # TODO: not implemented right now.

    compared_dir = args.compared_dir
    c_nsamples = args.c_nsamples

    reward_model = load_reward_model(reward_model_id)

    # if 'PairRM' in reward_model_id:
    #     assert compared_dir != "None"
    #     assert n_samples == 1

    src_lines = load_dataset(dataset) # src is used only by comet and clip.
    # trg_lines = load_dataset(dataset, ref=True)
    

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(reward_dir, dataset, model_n), exist_ok=True)
    os.makedirs(os.path.join('../reward_summary', dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))
    filtered_files = load_samples_from_file(files, epsilon, topk, topp, True, 0, 0.0)
    filtered_files.sort(key=lambda x: int(x.split('_')[0]))
    assert len(filtered_files) > 0

    if compared_dir != "None":
        compared_files = sorted(os.listdir(compared_dir))
        c_filtered_files = load_samples_from_file(compared_files, epsilon, topk, topp, True, 0, 0.0)
        c_filtered_files.sort(key=lambda x: int(x.split('_')[0]))

    print('first 10 files=', filtered_files[:10])
    print('n_files=', len(filtered_files))

    rows = []

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break
        filename = filtered_files[sample_id]
        assert "{:04}".format(sample_id) in filename

        if isinstance(src_lines[sample_id], list):
            src_input = src_lines[sample_id][0]["content"]
        else:
            src_input = src_lines[sample_id]

        df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]
        df.fillna("", inplace=True)
        hyp = list(df.iloc[:]['text'].astype(str))
        
        if compared_dir != "None":
            c_filename = c_filtered_files[sample_id]
            c_df = pd.read_csv(os.path.join(compared_dir, c_filename))
            c_df = c_df[:c_nsamples]
            c_hyps = c_df.iloc[:]['text'].astype(str).tolist()
            scores = [reward_model.get_winratio(src_input, hyp[0], c_hyps)]
        else:
            scores = reward_model.get_rewards(src_input, hyp)

        rows.append(scores)
        
        reward_model_n = os.path.basename(reward_model_id)

        filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}".format(sample_id, epsilon, topk, topp, reward_model_n)
        outfilepath = os.path.join(reward_dir, dataset, model_n, filename)

        df = pd.DataFrame(scores, columns=[reward_model_id])
        df.to_csv(outfilepath, index=False)

    df_summary = pd.DataFrame(rows)
    sum_filename = "eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}".format(epsilon, topk, topp, reward_model_n)
    sum_path = os.path.join('../reward_summary', dataset, model_n, sum_filename)
    df_summary.to_csv(sum_path, index=False)
