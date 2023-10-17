# A Tale of Pronouns

Code associated with the paper "A Tale of Pronouns: Interpretability Informs Gender Bias Mitigation for Fairer Instruction-Tuned Machine Translation".

## Getting Started

Beware, the operation might break existing venv/conda environments. We recommend working on a separate environment.
We conducted all our experiments with Python 3.10. To get started, install the requirements listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Dataset

To run our code, populate the `datasets` folder with the following files:

- winomt_en.txt (WinoMT)
- test2006.de (Europarl)
- test2006.en
- test2006.es

These files are publicly available. If you do not know where to find them, shoot us an email.

## Releases

This repository contains the following assets described in the paper:

- human-refined GPT-3.5 translation of WinoMT professions
- human-translated seed demonstrations for few-shot learning
- integrated gradient scores (WIP)

## Replicating Our Results

Many scripts require to specify a prompt template. See `./src/utils.py` the available options.

### Translating Europarl and WinoMT

```bash
./bash/translate_dataset.sh europarl-test es 0
./bash/translate_dataset.sh europarl-test de 0
./bash/translate_dataset.sh winomt es 0
./bash/translate_dataset.sh winomt de 0
```

### Evaluating Europarl Translations

```bash
./bash/evaluate_all.sh europarl-test ./translations/europarl-test/ en es
./bash/evaluate_all.sh europarl-test ./translations/europarl-test/ en de
```

### Evaluating WinoMT Translations

We use WinoMT's original code to evaluate delta_G, delta_S, and accuracy. We will provide a detail guide on that. Meanwhile, you can refer to the [official repository](https://github.com/gabrielStanovsky/mt_gender).

### Generating Integrated Gradients

```bash
./bash/compute_integrated_gradients.sh winomt es
```