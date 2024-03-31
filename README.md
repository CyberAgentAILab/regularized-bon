## Regularized Best-of-N

Implementation of Regularized Best-of-N (RBoN).


```
pip install -r requirements.txt
```

## Running RBoN

Running RBoN takes a multiple steps. 

1. First you generate a set of responses using sample.sh. We use the same set of samples generated for all the algorithms for fair comparison.
2. Compute Wasserstein distance and KL divergence using compute_wd.sh and compute_logprob.sh. 
3. Run mbr/compute_rbon.py to compute RBoN-WD and RBoN-KL.

You get the scores in csv file.

### Sampling candidates

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

### Computing RBoN
```
python3 mbr/compute_rbon.py --dataset alpaca --ncandidates [NUMBER OF SAMPLES]
```