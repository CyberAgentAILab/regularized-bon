# Default parameters are set to run a debug experiment.

DOMAIN=alpaca
MODEL=HuggingFaceH4/mistral-7b-sft-beta
NLINES=3
NSAMPLES=5
EPS=0.01
TOPK=0
TOPP=1.0
DEBUG=0
STARTITER=0

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h:c: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        # TODO: Long options
        REWARD=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute";;
    a)
        STARTITER=${OPTARG};;
    c)
        COMPARISON=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done


DATADIR=None


basemodel=`echo "${MODEL##*/}"`


# Return an error if the python script fails
set -e


python3 mbr/compute_logprob.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$basemodel \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP

if [ "$DEBUG" == "1" ]; then
    echo "done!"
fi

# Notification
MESSAGE="./experiments/run_logprob.sh $@"