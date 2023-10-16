import os

import pandas as pd


def read_data(dataset_name, src_lang, tgt_lang, return_data=False):
    """Read and parse dataset with source and target references"""

    if dataset_name == "winomt":
        data_file = "./datasets/winomt_en.txt"
        data = pd.read_csv(data_file, sep="\t")
        sources = data["text"].tolist()
        references = None

    # for europarl see https://www.statmt.org/wmt06/shared-task/
    elif dataset_name == "europarl-devtest":
        with open(f"./datasets/devtest2006.{src_lang}", encoding="latin-1") as fp:
            sources = [l.strip() for l in fp.readlines()]
        with open(f"./datasets/devtest2006.{tgt_lang}", encoding="latin-1") as fp:
            references = [l.strip() for l in fp.readlines()]

    elif dataset_name == "europarl-test":
        with open(f"./datasets/test2006.{src_lang}", encoding="latin-1") as fp:
            sources = [l.strip() for l in fp.readlines()]
        with open(f"./datasets/test2006.{tgt_lang}", encoding="latin-1") as fp:
            references = [l.strip() for l in fp.readlines()]

    else:
        raise ValueError("Dataset not known!")

    out = sources, references
    if return_data:
        out += data

    return sources, references


ISO_TO_LANG = {
    "es": "Spanish",
    "en": "English",
    "it": "Italian",
    "fr": "French",
    "de": "German",
}


def build_prompt(
    template_id: int,
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    file_few_shot: str = None,
):
    """Use this to test new prompts"""
    if template_id == 0:
        prompt = f"{src_text}\n\nTranslate this to {ISO_TO_LANG[tgt_lang]}?"
    elif template_id == 1:
        prompt = f"Translate from {ISO_TO_LANG[src_lang]} to {ISO_TO_LANG[tgt_lang]}:\n\n{src_text}\n\n{ISO_TO_LANG[tgt_lang]}:"
    elif template_id == 2:
        prompt = f"Translate the following sentence to {ISO_TO_LANG[tgt_lang]}:\n{src_text}\n\n{ISO_TO_LANG[tgt_lang]}:"
    elif template_id == 3:
        file_path = os.path.join("./results/seeds_for_fewshot", file_few_shot)
        df = pd.read_csv(file_path, sep="\t")
        inputs_prefix = "Q: "
        targets_prefix = "A: "
        x_y_delimiter = "\n\n"
        example_separator = "\n\n\n"

        prompt = ""
        for index, row in df.iterrows():
            src_text_few_shot = row["scr_txt"]
            target_text = row["sent_example"]

            inputs = f'Translate "{src_text_few_shot}" to {ISO_TO_LANG[tgt_lang]}?\n'
            targets = target_text + "\n"

            prompt += f"{inputs_prefix}{inputs}{x_y_delimiter}{targets_prefix}{targets}{example_separator}"

        inputs = f'Translate "{src_text}" to {ISO_TO_LANG[tgt_lang]}?\n'
        prompt += f"{inputs_prefix}{inputs}{x_y_delimiter}{targets_prefix}"

    elif template_id == 4:
        prompt = f"Translate the following text from {ISO_TO_LANG[src_lang]} to {ISO_TO_LANG[tgt_lang]} {src_text}"

    else:
        raise NotImplementedError("Wrong template id")
    return prompt
