# Default parameters are set to run a debug experiment.

DOMAIN=alpaca
MODEL=dolly-v2-3b
NLINES=3
NSAMPLES=5
EPS=0.01
TOPK=0
TOPP=1.0
REWARD=OpenAssistant/reward-model-deberta-v3-large-v2
DEBUG=0
STARTITER=0
COMPARISON=None

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
        TOPP=${OPTARG};;
    i)
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


if [ "$COMPARISON" == "None" ];
then
    CSAMPLES=0
    CDIR=None
else
    CSAMPLES=$NSAMPLES
    NSAMPLES=1
    CDIR=./samples/$DOMAIN/$COMPARISON
fi
DATADIR=None



# Return an error if the python script fails
set -e

python3 mbr/reward_engine.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$MODEL \
    --compared_dir $CDIR \
    --n_lines $NLINES --start_iter $STARTITER \
    --n_samples $NSAMPLES \
    --c_nsamples $CSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --reward_model $REWARD

if [ "$DEBUG" == "1" ]; then
    echo "done!"
fi

# Notification
MESSAGE="./experiments/run_reward.sh $@"
