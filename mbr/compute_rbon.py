import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def compute_scores(dataset, model, proxy, gold, alphas, ninstances=805, ncandidates=128):
    matrix_file = "matrix/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00_sentbert_" + str(ncandidates)
    prob_file = "logprob/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00"
    reward_file = "reward/{}/{}/".format(dataset, model) + "{:04d}_eps-0.01_topk-00_topp-1.00_{}"

    rows = []
    
    if proxy == gold:
        reward_names = [os.path.basename(proxy)]
    else:
        reward_names = [os.path.basename(proxy), os.path.basename(gold)]

    for instance_id in tqdm(range(ninstances)):
    # for instance_id in range(805):
        matrix_df = pd.DataFrame(np.loadtxt(matrix_file.format(instance_id))[:ncandidates,:ncandidates])
        rewards = []
        for i in range(len(reward_names)):
            reward_df = pd.read_csv(reward_file.format(instance_id, reward_names[i]))[:ncandidates]
            rewards.append(reward_df)
        reward_df = pd.concat(rewards, axis=1)

        df = pd.concat([matrix_df, reward_df], axis=1)

        logprob_df = pd.read_csv(prob_file.format(instance_id))[:ncandidates]

        # BoN sampling
        bon_idx = df[proxy].argmax()
        bon_score = df.iloc[bon_idx][gold]

        # MBR decoding
        mbr_idx = df[matrix_df.columns].sum(axis=1).argmax()
        mbr_score = df.iloc[mbr_idx][gold]

        # WD-Regularized BoN
        wdrbon_scores = []
        for alpha in alphas:
            mbr_bon = (df[proxy] + alpha * df[matrix_df.columns].mean(axis=1)).argmax()
            mbr_bon_score = df.iloc[mbr_bon][gold]
            wdrbon_scores.append(mbr_bon_score)

        # KL-Regularized BoN
        klbon_scores = []
        for alpha in alphas:
            kl_bon = (df[proxy] + alpha * logprob_df[model]).argmax()
            kl_bon_score = df.iloc[kl_bon][gold]
            klbon_scores.append(kl_bon_score)

        # The Oracle score for debugging
        oracle_idx = df[gold].argmax()
        oracle_score = df.iloc[oracle_idx][gold]

        # The average score (score of random sampling)
        avg_score = df[gold].mean()

        row = [bon_score, mbr_score] + wdrbon_scores + klbon_scores + [oracle_score, avg_score]
        rows.append(row)
    df = pd.DataFrame(rows,
        columns=['BoN', 'MBR'] + ['WD-BoN-{}'.format(alpha) for alpha in alphas] + ['KL-BoN-{}'.format(alpha) for alpha in alphas] + ['Oracle', 'Random'])

    df.to_csv('results/klbon_' + os.path.basename(proxy) + "_" + os.path.basename(gold) + "_" + str(ncandidates) + ".csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='alpaca')
    parser.add_argument("--model", type=str, default='gpt2')
    parser.add_argument("--proxy", type=str, default='stanfordnlp/SteamSHP-flan-t5-large')
    parser.add_argument("--gold", type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2')
    parser.add_argument("--ninstances", type=int, default=4)
    parser.add_argument("--ncandidates", type=int, default=4)

    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    proxy = args.proxy
    gold = args.gold
    ninstances = args.ninstances
    ncandidates = args.ncandidates

    alphas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    os.makedirs('./results', exist_ok=True)
    compute_scores(dataset, model, proxy, gold, alphas, ninstances=ninstances, ncandidates=ncandidates)
