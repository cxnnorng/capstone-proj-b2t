#SBATCH --job-name=b2txt25_eval_monophones
#SBATCH --account=connorng-2
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

#SBATCH --output=b2txt25_eval_monophones_%j.out
#SBATCH --error=b2txt25_eval_monophones_%j.err

# Start Redis in background
~/redis-7.2.4/src/redis-server --daemonize yes

# Activate LM environment
conda activate b2txt25_lm

# Start language model in background
python language_model/language-model-standalone.py \
  --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
  --do_opt \
  --nbest 100 \
  --acoustic_scale 0.325 \
  --blank_penalty 90 \
  --alpha 0.55 \
  --redis_ip localhost \
  --gpu_number 0 &

# Switch env
conda activate b2txt25

# Run evaluation
python evaluate_model.py \
  --model_path ../data/t15_pretrained_rnn_baseline \
  --data_dir ../data/hdf5_data_final \
  --eval_type test \
  --gpu_number 0

# Shutdown Redis
~/redis-7.2.4/src/redis-cli shutdown