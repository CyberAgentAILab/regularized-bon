import os
import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import torch

from utility_func import *
from utils import load_model, load_dataset, load_samples_from_file, result_dir, matrix_dir, reward_dir
from parser import get_mbr_parser


def compute_logprob(tokenizer, instruction, response):

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]

    concatenated = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    response_len = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt").shape[1]

    with torch.no_grad():
        outputs = model(concatenated)
        logit_predictions = outputs.logits
    response_logit = logit_predictions[0, -response_len, :]
    response_log_probs = torch.nn.functional.log_softmax(response_logit, dim=-1)

    log_sum = 0
    for i in range(response_len):
        log_token_prob = response_log_probs[concatenated[0, -response_len + i]].item()
        log_sum += log_token_prob
    

    return log_sum


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

    quantize = args.quantize

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    src_lines = load_dataset(dataset)


    tokenizer, model, model_name, stop_tokens = load_model(dataset, torch_device, model_name, quantize)

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join('./logprob', dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))
    filtered_files = load_samples_from_file(files, epsilon, topk, topp, True, 0, 0.0)
    filtered_files.sort(key=lambda x: int(x.split('_')[0]))
    assert len(filtered_files) > 0

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

        probs = []
        for i, outputs in enumerate(hyp):
            p = compute_logprob(tokenizer, src_input, outputs)
            probs.append(p)
        
        filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(sample_id, epsilon, topk, topp)
        outfilepath = os.path.join('./logprob', dataset, model_n, filename)

        df = pd.DataFrame(probs, columns=[model_n])
        df.to_csv(outfilepath, index=False)
