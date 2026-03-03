#!/bin/bash
#SBATCH --job-name=b2txt25_eval_monophones
#SBATCH --account=connorng-2
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

#SBATCH --output=b2txt25_eval_monophones_%j.out
#SBATCH --error=b2txt25_eval_monophones_%j.err

module load Miniconda3/25.11.1-1

# Start Redis in background
# ~/redis-7.2.4/src/redis-server --daemonize yes
~/redis-7.2.4/src/redis-server &
REDIS_PID=$!

# Wait for Redis to start
until ~/redis-7.2.4/src/redis-cli ping; do
  sleep 1
done

# Activate LM environment
source activate b2txt25_lm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Start language model in background
python ../language_model/language-model-standalone.py \
  --lm_path ../language_model/pretrained_language_models/speech_5gram/lang_test \
  --do_opt \
  --nbest 100 \
  --acoustic_scale 0.325 \
  --blank_penalty 90 \
  --alpha 0.55 \
  --redis_ip localhost \
  --gpu_number 0 &

LM_PID=$!

echo "Waiting up to 60 minutes for LM to connect..."

for i in {1..3600}; do
  if ~/redis-7.2.4/src/redis-cli info clients | grep -q "connected_clients:2"; then
    echo "LM connected to Redis."
    break
  fi
  sleep 1
done

# Switch env
source activate b2txt25

# Run evaluation
python evaluate_model.py \
  --model_path ../model_training/trained_models/baseline_rnn \
  --data_dir ../data/hdf5_data_final \
  --eval_type val \
  --gpu_number 0

# Shutdown Redis
# ~/redis-7.2.4/src/redis-cli shutdown

kill $LM_PID
kill $REDIS_PID