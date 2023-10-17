#!/bin/bash
#SBATCH --job-name=mt_eval_all
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000MB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm-%A.out

source /home/AttanasioG/.bashrc
conda activate py310
export TOKENIZERS_PARALLELISM=true

DATASET_NAME=$1
INPUT_DIR=$2
SRC_LANG=$3
TGT_LANG=$4

if [ -z $DATASET_NAME ]; then
    echo "specify a dataset"
    exit -1
fi

if [ -z $INPUT_DIR ]; then
    echo "specifiy an input dir"
    exit -1
fi

MODELS=( \
    "Helsinki-NLP/opus-mt-${SRC_LANG}-${TGT_LANG}" \
    "google/flan-t5-small" \
    "google/flan-t5-base" \
    "google/flan-t5-large" \
    "google/flan-t5-xl" \
    "google/flan-t5-xxl" \
    "bigscience/mt0-small" \
    "bigscience/mt0-base" \
    "bigscience/mt0-large" \
    "bigscience/mt0-xl" \
    "bigscience/mt0-xxl" \
)

for model in ${MODELS[@]}; do

    echo "Evaluating ${model}..."

    python src/evaluate_metrics.py \
        --input_dir=${INPUT_DIR} \
        --model_name_or_path=${model} \
        --dataset_name=${DATASET_NAME} \
        --src_lang=${SRC_LANG} \
        --tgt_lang=${TGT_LANG}
done