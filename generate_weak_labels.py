"""Generate and evaluate LLM-based weak-label rules."""

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="diabetes", choices=utils.TASK_DICT)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--function-model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--num-conditions", type=int, default=20)
    parser.add_argument("--condition-batch-size", type=int, default=10)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="evaluate existing artifacts without making API calls",
    )
    return parser.parse_args()


def load_dataset(name, seed):
    df, _, test, _, _, target, labels, is_categorical = utils.get_dataset(
        name, seed, shot=0
    )
    features = df.drop(columns=target)
    answers = np.array([labels.index(value) for value in df[target]])

    if name in {"communities", "myocardial"}:
        with open(f"data/{name}_selected_column_indices.pkl", "rb") as file:
            selected = pickle.load(file)
        features = features.iloc[:, selected]
        test = test.iloc[:, selected]
        is_categorical = np.asarray(is_categorical)[selected].tolist()

    return features, test, answers, labels, is_categorical


def format_examples(features, indices):
    blocks = []
    for number, index in enumerate(indices, start=1):
        blocks.append(f"Sample {number}:\n{features.loc[index].to_string()}")
    return "\n\n".join(blocks) + "\n"


def extract_function(response):
    try:
        return response.split("<start>", 1)[1].split("<end>", 1)[0].strip()
    except IndexError as error:
        raise ValueError("function response lacks <start>/<end> markers") from error


def artifact_paths(args, trial):
    directory = Path(f"conditions/0_shot/{trial}_trial/{args.model}")
    stem = f"{args.dataset}_{args.num_conditions}_{args.seed}_{args.num_examples}"
    return directory / f"{stem}_conditions.pkl", directory / f"{stem}_condition2index.pkl"


def generate(args, features, labels, feature_description):
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key to generate artifacts.")

    label_description = "[" + ",".join(map(str, labels)) + "]"
    condition_template = "templates/generate_conditions.txt"
    function_template = "templates/ask_for_function.txt"

    for trial in range(args.num_trials):
        rng = random.Random(args.seed + trial)
        all_conditions = [[] for _ in labels]
        all_descriptions = [[] for _ in labels]
        all_indices = defaultdict(list)

        for batch in range(1, args.max_batches + 1):
            if all(len(all_indices[label]) >= args.num_conditions for label in range(len(labels))):
                break

            examples = format_examples(
                features, rng.sample(list(features.index), args.num_examples)
            )
            prompt = utils.get_prompt(condition_template, {
                "[NUM_CONDITION]": str(args.condition_batch_size),
                "[TASK]": utils.TASK_DICT[args.dataset],
                "[FEATURES]": feature_description,
                "[EXAMPLES]": examples,
                "[LABEL_LIST]": label_description,
            })
            response = utils.query_gpt(
                prompt, args.api_key, temperature=0, model=args.model
            ).replace("\n\n", "\n")
            conditions, descriptions = utils.parse_conditions(response, labels, "Label")

            batch_indices = defaultdict(list)
            for label, label_conditions in enumerate(conditions):
                iterator = tqdm(label_conditions, desc=f"trial {trial} batch {batch} label {label}")
                for condition_number, condition in enumerate(iterator):
                    prompt = utils.get_prompt(function_template, {
                        "[NAME]": f"condition_{label}_{condition_number}",
                        "[FEATURES]": feature_description,
                        "[CONDITIONS]": condition,
                    })
                    response = utils.query_gpt(
                        prompt, args.api_key, temperature=0, model=args.function_model
                    )
                    try:
                        indices = utils.run_function(extract_function(response), features)
                    except ValueError:
                        continue
                    if isinstance(indices, list):
                        batch_indices[label].append((condition_number, sorted(indices)))

            for label in range(len(labels)):
                for condition_number, indices in batch_indices[label]:
                    if indices and indices not in all_indices[label]:
                        all_indices[label].append(indices)
                        all_conditions[label].append(conditions[label][condition_number])
                        all_descriptions[label].append(descriptions[label][condition_number])

            counts = [len(all_indices[label]) for label in range(len(labels))]
            print(f"trial={trial} batch={batch} rule_counts={counts}")
        if not all(len(all_indices[label]) >= args.num_conditions for label in range(len(labels))):
            raise RuntimeError(
                f"failed to collect {args.num_conditions} rules per label "
                f"within {args.max_batches} batches"
            )

        for label in range(len(labels)):
            all_indices[label] = all_indices[label][:args.num_conditions]
            all_conditions[label] = all_conditions[label][:args.num_conditions]
            all_descriptions[label] = all_descriptions[label][:args.num_conditions]

        conditions_path, indices_path = artifact_paths(args, trial)
        conditions_path.parent.mkdir(parents=True, exist_ok=True)
        with conditions_path.open("wb") as file:
            pickle.dump([all_conditions, all_descriptions], file)
        with indices_path.open("wb") as file:
            pickle.dump(all_indices, file)
        print(f"saved {conditions_path}\nsaved {indices_path}")


def evaluate(args, features, test, answers, labels):
    results = []
    multiclass = len(labels) > 2
    for trial in range(args.num_trials):
        _, indices_path = artifact_paths(args, trial)
        with indices_path.open("rb") as file:
            condition_indices = pickle.load(file)
        weak_labels = utils.generate_feature(
            condition_indices, len(features), len(labels), args.num_conditions
        )
        results.append(utils.evaluate(
            weak_labels, test.index, answers, multiclass,
            len(labels), args.num_conditions,
        ))

    values = np.asarray(results)
    print("trials:", " ".join(f"{value:.4f}" for value in values))
    print(
        f"mean={values.mean():.4f} std={values.std():.4f} "
        f"min={values.min():.4f} max={values.max():.4f}"
    )


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    features, test, answers, labels, is_categorical = load_dataset(
        args.dataset, args.seed
    )

    if not args.evaluate_only:
        feature_description = utils.get_feature_desc(
            features, is_categorical, f"data/{args.dataset}-metadata.json"
        )
        generate(args, features, labels, feature_description)

    evaluate(args, features, test, answers, labels)


if __name__ == "__main__":
    main()
