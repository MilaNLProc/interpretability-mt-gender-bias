#!/bin/bash
#SBATCH --job-name=ig_europarl
#SBATCH --gpus=1
#SBATCH --ntasks=4
#SBATCH --mem=128000MB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuseppe.attanasio3@unibocconi.it

export TOKENIZERS_PARALLELISM=true

source /home/AttanasioG/.bashrc
conda activate py310

DATASET_NAME=$1
TGT_LANG=$2
RESULTS_DIR=$3

if [ -z $DATASET_NAME ]; then
    echo "specify a dataset"
    exit -1
fi

if [ -z $RESULTS_DIR ]; then
    echo "Main results dir is not specified, setting to ./feature_attributions"
    RESULTS_DIR="./feature_attributions"
fi

MODELS=( \
    "google/flan-t5-xxl"
    "bigscience/mt0-xxl" \
)
DECODING_STRATEGY="beam_search"
PROMPT_TEMPLATE="0"

for model in ${MODELS[@]}; do
    
    OUTPUT_DIR="${RESULTS_DIR}/integrated_gradients/${DECODING_STRATEGY}/prompt_${PROMPT_TEMPLATE}/${model}"

    python src/run.py --do_translation="false" --do_feature_attribution="true" \
        --model_name_or_path=${model} \
        --dataset_name=${DATASET_NAME} \
        --src_lang="en" --tgt_lang=${TGT_LANG} \
        --output_dir=${OUTPUT_DIR} \
        --batch_size="16" \
        --ig_steps="32" --ig_internal_batch_size="32" \
        --prompt_template=${PROMPT_TEMPLATE} \
        --num_return_sequences="1" \
        --max_new_tokens="1024" \
        --num_beams="4" \
        --early_stopping="true" \
        --quantization="8b"

done
