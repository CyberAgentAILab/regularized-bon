## Regularized Best-of-N

Implementation of [Regularized Best-of-N (RBoN)](https://openreview.net/forum?id=ewRlZPAReR).

The code is tested on Ubuntu 20.04 using Python 3.8 and CUDA 11.0 (Docker image nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04).


```
git clone git@github.com:CyberAgentAILab/regularized-bon
cd regularized-bon
pip install -r requirements.txt
```

## Usage

Running RBoN takes multiple steps. 

1. First you generate a set of responses using sample.sh. We use the same set of samples generated for all the algorithms for fair comparison.
2. Compute Wasserstein distance and KL divergence using compute_wd.sh and compute_logprob.sh. 
3. Compute the reward of the responses.
3. Run mbr/compute_rbon.py to compute MBR-BoN (RBoN-WD) and RBoN-KL.

You get the CSV file in the results/ directory.

### Sampling candidates

By default, it runs using [openai-community/gpt2](https://huggingface.co/openai-community/gpt2). Add `-m [MODEL NAME IN HUGGINGFACE HUB]` to change the language model.

```
./experiments/sample.sh -d alpaca -s [NUMBER OF SAMPLES] 
```

### Computing Wasserstein distance

```
./experiments/compute_wd.sh -d alpaca -s [NUMBER OF SAMPLES] 
```

### Computing log probability

```
./experiments/compute_logprob.sh -d alpaca -s [NUMBER OF SAMPLES] 
```

### Computing the reward of the samples

```
./experiments/compute_reward.sh -d alpaca -s [NUMBER OF SAMPLES] -i stanfordnlp/SteamSHP-flan-t5-large
./experiments/compute_reward.sh -d alpaca -s [NUMBER OF SAMPLES] -i OpenAssistant/reward-model-deberta-v3-large-v2
```


### Computing MBR-BoN and RBoN_KL
```
python3 mbr/compute_rbon.py --dataset alpaca --ncandidates [NUMBER OF SAMPLES]
```


## Reference

Jinnai, Y., Morimura, T., Ariu, K., and Abe, K. Regularized Best-of-N Sampling with Minimum Bayes Risk Objective for Language Model Alignment. 2025.

Bibtex:
```
@misc{jinnai2025regularizedbestofnsamplingminimum,
      title={Regularized Best-of-N Sampling with Minimum Bayes Risk Objective for Language Model Alignment}, 
      author={Yuu Jinnai and Tetsuro Morimura and Kaito Ariu and Kenshi Abe},
      year={2025},
      eprint={2404.01054},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.01054}, 
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.
