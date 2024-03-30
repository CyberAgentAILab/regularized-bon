# Default parameters are set to run a debug experiment.

DOMAIN=alpaca
MODEL=mistral-7b-sft-beta
NLINES=3
NSAMPLES=5
EPS=0.01
TOPK=0
TOPP=1.0
SIM=sentbert
DEBUG=0
RECOMPUTE=""

DOSAMPLE=1

STARTITER=0

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h:c: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        TOPP=${OPTARG};;
    i)
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute_matrix";;
    c)
        STARTITER=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done
DATADIR=None

# Return an error if the python script fails
set -e

python3 mbr/compute_wd.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$MODEL \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM