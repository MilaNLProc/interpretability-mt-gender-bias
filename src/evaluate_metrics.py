import glob
import json
import os
from dataclasses import asdict, dataclass, field

import evaluate
import numpy as np
from codecarbon import track_emissions
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from transformers import HfArgumentParser

from utils import read_data


@dataclass
class Arguments:
    dataset_name: str = field()
    model_name_or_path: str = field()
    src_lang: str = field()
    tgt_lang: str = field()
    input_dir: str = field()


def comet_score(sources, references, translations, model_name, reference_free=False):
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    if reference_free:
        data = [{"src": s, "mt": t} for s, r, t in zip(sources, references)]
    else:
        data = [
            {"src": s, "mt": t, "ref": r}
            for s, r, t in zip(sources, references, translations)
        ]

    model_output = model.predict(data, batch_size=128, gpus=1)
    return model_output


def evaluate_metrics(sources, references, translations, tgt_lang):
    assert len(sources) == len(references)
    assert len(references) == len(translations)

    metrics = dict()

    # COMET metrics
    comet_22 = comet_score(sources, references, translations, "Unbabel/wmt22-comet-da")
    metrics["wmt22-comet-da"] = comet_22["system_score"]
    comet_22 = comet_score(
        sources, references, translations, "Unbabel/wmt20-comet-qe-da"
    )
    metrics["wmt20-comet-qe-da"] = comet_22["system_score"]

    # BERTscore metrics
    bertscore = evaluate.load("bertscore")
    bscores = bertscore.compute(
        predictions=translations, references=references, lang=tgt_lang
    )
    metrics["bertscore_f1"] = np.mean(bscores["f1"])
    metrics["bertscore_precision"] = np.mean(bscores["precision"])
    metrics["bertscore_recall"] = np.mean(bscores["recall"])

    # BLEU metrics
    bleu = evaluate.load("bleu")

    metrics["bleu-4"] = bleu.compute(predictions=translations, references=references)[
        "bleu"
    ]
    metrics["bleu-2"] = bleu.compute(
        predictions=translations, references=references, max_order=2
    )["bleu"]

    return metrics


@track_emissions
def main():
    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    model_base_name = os.path.basename(args.model_name_or_path)

    print("Arguments", args)
    sources, references = read_data(args.dataset_name, args.src_lang, args.tgt_lang)

    files_to_evaluate = glob.glob(
        f"{args.input_dir}/**/*{model_base_name}_{args.dataset_name}_{args.src_lang}_{args.tgt_lang}.json",
        recursive=True,
    )
    print("Files to evaluate (Count):", len(files_to_evaluate))

    for file in tqdm(files_to_evaluate, desc="Files to evaluate"):
        basename = os.path.basename(file).split(".")[0]
        dirname = os.path.dirname(file)
        new_output_file = f"{dirname}/{basename}_metrics.json"

        if os.path.exists(new_output_file):
            print(f"File {new_output_file} exists alread. Skipping...")
            continue

        with open(file) as fp:
            results_dict = json.load(fp)

        translations = results_dict["translations"]

        # Run the actual evaluation
        metrics = evaluate_metrics(sources, references, translations, args.tgt_lang)
        print(metrics)

        # Save results in a file with similar format
        results_dict.pop("references")
        results_dict.pop("translations")

        results_dict |= metrics
        with open(new_output_file, "w") as fp:
            json.dump(results_dict, fp, indent=4)


if __name__ == "__main__":
    main()
