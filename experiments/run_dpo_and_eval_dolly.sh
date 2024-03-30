set -e

#############################
# Run DPO
#############################
./experiments/run_dpo.sh "$@"

MODELNAME=`cat /data/model_name.txt`
echo "Running experiments for $MODELNAME"

#############################
# Evaluate the model
#############################
./experiments/sample.sh -d alpaca -m $USERNAME/$MODELNAME -s 1 -l 805  -z 1 -e 0.0 -n 0.9 -r -p dummy.txt

./experiments/run_reward.sh -d alpaca -m $MODELNAME -s 1 -l 805 -e 0.0 -n 0.9 

# TODO: This compares against dolly.
./experiments/run_reward.sh -d alpaca -m $MODELNAME -c dolly-v2-3b -l 805 -s 32 -e 0.0 -n 0.9 -i llm-blender/PairRM
