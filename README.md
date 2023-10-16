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
