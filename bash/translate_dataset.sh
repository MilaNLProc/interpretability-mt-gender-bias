#!/bin/bash
#!/bin/bash
#SBATCH --job-name=mt_dataset
#SBATCH --gpus=1
#SBATCH --ntasks=4
#SBATCH --mem=64000MB
#SBATCH --time=24:00:00
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuseppe.attanasio3@unibocconi.it

source /home/AttanasioG/.bashrc
conda activate py310

export TOKENIZERS_PARALLELISM=true

DATASET_NAME=$1
TGT_LANG=$2
PROMPT_TEMPLATE=$3

if [ -z $DATASET_NAME ]; then
    echo "specify a dataset"
    exit -1
fi
if [ -z $TGT_LANG ]; then
    echo "specify a tgt lang"
    exit -1
fi

MODELS=( \
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

echo "Running MarianMT baseline..."

python src/run.py \
    --model_name_or_path="Helsinki-NLP/opus-mt-en-${TGT_LANG}" \
    --dataset_name=${DATASET_NAME} \
    --src_lang="en" --tgt_lang="${TGT_LANG}" \
    --output_dir="./translations/${DATASET_NAME}/opus-mt-en-${TGT_LANG}" \
    --num_beams=4 \
    --batch_size=128

for model in ${MODELS[@]}; do
    IFS='/'
    read -a model_toks <<< "${model}"
    IFS=' '
    echo "Translating with ${model_toks[1]}..."

    python src/run.py \
        --model_name_or_path=${model} \
        --dataset_name=${DATASET_NAME} \
        --src_lang="en" --tgt_lang="${TGT_LANG}" \
        --output_dir="./translations/${DATASET_NAME}/${model_toks[1]}" \
        --batch_size=128 \
        --num_beams=4 \
        --prompt_template=${PROMPT_TEMPLATE}

done