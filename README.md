## Regularized Best-of-N

Implementation of Regularized Best-of-N (RBoN).


```
pip install -r requirements.txt
```

## Running RBoN

Running RBoN takes a multiple steps. First you generate a set of responses using sample.sh. We use the same set of samples generated for all the algorithms for fair comparison. compute_wd.sh computes the Wasserstein distance for RBoN-WD and compute_logprob.sh computes the log probability used in RBoN-KL. Then finally we run mbr/compute_rbon.py to compute the result of the decoding.

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