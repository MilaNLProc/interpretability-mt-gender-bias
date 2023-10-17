import json
import logging
import os
import pdb
import pickle
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from simple_generation import SimpleGenerator
from tqdm import tqdm
from transformers import HfArgumentParser

from utils import ISO_TO_LANG, build_prompt, read_data

logger = logging.getLogger(__name__)

# load and env variable in .env
load_dotenv()


@dataclass
class Arguments:
    model_name_or_path: str = field()
    dataset_name: str = field()
    src_lang: str = field()
    tgt_lang: str = field()
    batch_size: int = field()
    file_few_shot: str = field(default=None)
    do_translation: bool = field(default=True)
    do_feature_attribution: bool = field(default=False)
    output_dir: str = field(default=None)
    use_default_generation: bool = field(default=False)
    prompt_template: int = field(default=None)
    dry_run: bool = field(default=False)
    overwrite_results: bool = field(default=False)
    ig_steps: int = field(default=16)
    ig_internal_batch_size: int = field(default=4)
    quantization: str = field(default="8b")
    lora_weights: str = field(default=None)
    start_batch_idx: int = field(default=0)


# see https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
@dataclass
class GenerationArguments:
    """Dataclass to choose the generation params.

    We follow HF defaults (see link above), but they can be overriden as command line paramters when calling the script.
    """

    num_return_sequences: int = field(default=1)
    max_new_tokens: int = field(default=1024)
    num_beams: int = field(default=1)
    early_stopping: bool = field(default=True)
    top_k: int = field(default=0)
    do_sample: bool = field(default=False)
    top_p: float = field(default=1)  # nucleus sampling
    temperature: float = 1.0
    penalty_alpha: float = 0.0


def translate(
    texts,
    model_name_or_path,
    tgt_lang,
    batch_size,
    generation_args,
    lora_weights=None,
):
    """Translate a list of texts using a model."""
    model_args = dict(model_name_or_path=model_name_or_path)

    if "opus" not in model_name_or_path:
        model_args.update(
            dict(
                device_map="auto",
                load_in_8bit=True,
                lora_weights=lora_weights,
            )
        )

    generator = SimpleGenerator(**model_args)
    translations = generator(texts, **generation_args, starting_batch_size=batch_size)
    return translations


def main():
    parser = HfArgumentParser((Arguments, GenerationArguments))
    args, generation_args = parser.parse_args_into_dataclasses()
    logger.info(args)

    if args.do_translation == args.do_feature_attribution:
        raise ValueError("You can either do translation or feature attribution.")

    model_base_name = os.path.basename(args.model_name_or_path)

    few_shot_name = ""
    if args.file_few_shot is not None:
        few_shot_name = args.file_few_shot.split(".")[0]
        out_filename = f"{model_base_name}_{args.dataset_name}_{args.src_lang}_{args.tgt_lang}_{few_shot_name}.json"
    else:
        out_filename = f"{model_base_name}_{args.dataset_name}_{args.src_lang}_{args.tgt_lang}.json"

    if os.path.exists(f"{args.output_dir}/{out_filename}"):
        if not args.overwrite_results:
            print(
                "Results file exists, skipping... Set overwrite_results to overwrite it."
            )
            return

    otexts, references = read_data(args.dataset_name, args.src_lang, args.tgt_lang)

    if args.dry_run:
        otexts = otexts[:100]

    if args.prompt_template is not None:
        texts = [
            build_prompt(
                args.prompt_template,
                t,
                args.src_lang,
                args.tgt_lang,
                args.file_few_shot,
            )
            for t in otexts
        ]
    else:
        texts = otexts

    # Generate translations and insert them into the DataFrame
    if args.do_translation:
        logger.info("Starting Translation...")

        translations = translate(
            texts,
            args.model_name_or_path,
            ISO_TO_LANG[args.tgt_lang],
            args.batch_size,
            asdict(generation_args),
            args.lora_weights,
        )

        # Save outputs
        results = asdict(args)
        results.update(asdict(generation_args))
        results["references"] = references
        results["translations"] = translations

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

            with open(f"{args.output_dir}/{out_filename}", "w") as fp:
                json.dump(results, fp, ensure_ascii=False, indent=4)

            # if we are translating winomt save in the expected format
            if args.dataset_name == "winomt":
                with open(
                    f"{args.output_dir}/{args.src_lang}-{args.tgt_lang}.txt",
                    "w",
                ) as fp:
                    for s, t in zip(otexts, translations):
                        fp.write(f"{s} ||| {t}\n")

        print("Ending translation...")

    if args.do_feature_attribution:
        import inseq
        import numpy as np

        print("Starting Feature Attribution")
        model_kwargs = {"device_map": "auto"}
        if args.quantization == "8b":
            model_kwargs["load_in_8bit"] = True
            print("Setting quantization to 8 bit!")
        elif args.quantization == "4b":
            model_kwargs["load_in_4bit"] = True
            print("Setting quantization to 4 bit!")
        else:
            raise ValueError("Quantization can be either 8b or 4b")

        model = inseq.load_model(
            args.model_name_or_path, "integrated_gradients", model_kwargs=model_kwargs
        )

        # raise NotImplementedError()
        n_batches = len(texts) // args.batch_size
        print("Splitting texts into n batches", n_batches)
        batches = np.array_split(texts, n_batches)

        print(f"Creating {args.output_dir} if it doesn't exist.")
        os.makedirs(args.output_dir, exist_ok=True)

        for idx, batch in tqdm(enumerate(batches), desc="Batch", total=len(batches)):
            if os.path.exists(
                f"{args.output_dir}/{args.src_lang}-{args.tgt_lang}_gen_texts_{idx}-{few_shot_name}.txt"
            ):
                print(
                    f"Skipping batch {idx}... Beware that this works only if batch_size remained unchanged!"
                )
                continue

            if args.start_batch_idx > 0 and idx < args.start_batch_idx:
                print(f"Skipping batch {idx} (start id={args.start_batch_idx})...")
                continue

            out = model.attribute(
                input_texts=batch.tolist(),
                n_steps=args.ig_steps,
                return_convergence_delta=True,
                step_scores=["probability"],
                batch_size=len(batch),
                internal_batch_size=args.ig_internal_batch_size,
                generation_args=asdict(generation_args),
                show_progress=True,
            )

            # Saving the generations produced in the batch, one per line
            with open(
                f"{args.output_dir}/{args.src_lang}-{args.tgt_lang}_gen_texts_{idx}_{few_shot_name}.txt",
                "w",
            ) as fp:
                for g in out.info["generated_texts"]:
                    fp.write(f"{g}\n")

            with open(
                f"{args.output_dir}/{args.src_lang}-{args.tgt_lang}_ig_attr_{idx}.pkl",
                "wb",
            ) as fp:
                pickle.dump(out.sequence_attributions, fp)


if __name__ == "__main__":
    main()
