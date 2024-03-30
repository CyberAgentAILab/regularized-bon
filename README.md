## Regularized Best-of-N

Implementation of Regularized Best-of-N.


```
pip install requirements.txt
```

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing Wasserstein distance

```
./experiments/compute_wd.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing log probability

```
./experiments/compute_logprob.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing RBoN
```
python3 mbr/compute_rbon.py --dataset [DATASET] --ncandidates [NUMBER OF SAMPLES]
```