#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=2
#SBATCH --output=b0.2_lr1e-5_l0.0002_bs200.out
#SBATCH --job-name=kba_rl_combotest
#SBATCH --mem=32000
#SBATCH --nodes=1

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "assigned $NUM_GPUS gpu(s)"

# load modules
module load anaconda
module load cuda

source activate YINGMA_KB

cd /lustre/fs0/home/oburns/yingma_research_files/
pwd
echo "Starting Training:"
python -m code.model.trainer --base_output_dir "output/fb15k-237/" --path_length 3 --hidden_size 50 --embedding_size 50 --batch_size 200 --beta 0.2 --Lambda 0.0002 --use_entity_embeddings 0 --train_entity_embeddings 0 --train_relation_embeddings 1 --data_input_dir "datasets/data_preprocessed/FB15K-237/" --vocab_dir "datasets/data_preprocessed/FB15K-237/vocab" --model_load_dir "saved_models/fb15k-237" --load_model 0 --nell_evaluation 0 --label_gen 0 --learning_rate 1e-5 --total_iterations=2000 --hp_type "combo trials" --hp_level "b0.2 lr1e-5 l0.0002 bs200"

echo "Ending script..."
