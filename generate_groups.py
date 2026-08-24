"""Generate and evaluate LLM-based semantic hyperedges."""

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
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--group-batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--num-conditions", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=100)
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


def artifact_paths(args, num_groups):
    directory = Path("groups") / args.model
    stem = f"{args.dataset}_{args.group_batch_size}_{num_groups}_{args.num_examples}"
    return directory / f"{stem}_groups.pkl", directory / f"{stem}_indices.pkl"


def generate(args, features, labels, feature_description, num_groups):
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key to generate artifacts.")

    label_description = "[" + ",".join(map(str, labels)) + "]"
    group_template = "templates/generate_groups.txt"
    function_template = "templates/ask_for_function.txt"
    group_conditions, group_titles, groups = [], [], []

    for batch in range(args.max_batches):
        if len(groups) >= num_groups:
            break

        rng = random.Random(args.seed + batch)
        examples = format_examples(
            features, rng.sample(list(features.index), args.num_examples)
        )
        prompt = utils.get_prompt(group_template, {
            "[NUM_GROUP]": str(args.group_batch_size),
            "[TASK]": utils.TASK_DICT[args.dataset],
            "[FEATURES]": feature_description,
            "[EXAMPLES]": examples,
            "[LABEL_LIST]": label_description,
        })
        response = utils.query_gpt(
            prompt, args.api_key, temperature=0, model=args.model
        )
        try:
            titles, conditions = utils.parse_groups(response, "Group")
        except (IndexError, ValueError):
            print(f"batch={batch + 1} group parsing failed")
            continue

        new_groups = []
        for index, condition in enumerate(tqdm(conditions, desc=f"batch {batch + 1}")):
            prompt = utils.get_prompt(function_template, {
                "[FEATURES]": feature_description,
                "[DESCRIPTION]": titles[index],
                "[CONDITIONS]": " AND ".join(condition),
            })
            response = utils.query_gpt(
                prompt, args.api_key, temperature=0, model=args.function_model
            )
            try:
                indices = utils.run_function(extract_function(response), features)
            except ValueError:
                indices = None
            new_groups.append(indices)

        for title, condition, indices in zip(titles, conditions, new_groups):
            if isinstance(indices, list) and len(indices) > 1:
                group_titles.append(title)
                group_conditions.append(condition)
                groups.append(indices)

        print(f"batch={batch + 1} groups={len(groups)}/{num_groups}")
    if len(groups) < num_groups:
        raise RuntimeError(
            f"failed to collect {num_groups} groups within {args.max_batches} batches"
        )

    group_titles = group_titles[:num_groups]
    group_conditions = group_conditions[:num_groups]
    groups = groups[:num_groups]
    descriptions_path, indices_path = artifact_paths(args, num_groups)
    descriptions_path.parent.mkdir(parents=True, exist_ok=True)
    with descriptions_path.open("wb") as file:
        pickle.dump([group_conditions, group_titles], file)
    with indices_path.open("wb") as file:
        pickle.dump(groups, file)
    print(f"saved {descriptions_path}\nsaved {indices_path}")


def evaluate(args, features, test, answers, labels, num_groups):
    _, indices_path = artifact_paths(args, num_groups)
    with indices_path.open("rb") as file:
        groups = pickle.load(file)
    incidence = utils.generate_hypergraph(len(features), groups)
    multiclass = len(labels) > 2
    results = defaultdict(list)

    for trial in range(args.num_trials):
        path = Path(
            f"conditions/0_shot/{trial}_trial/{args.model}/"
            f"{args.dataset}_{args.num_conditions}_{args.seed}_{args.num_examples}_condition2index.pkl"
        )
        with path.open("rb") as file:
            condition_indices = pickle.load(file)
        weak_labels = utils.generate_feature(
            condition_indices, len(features), len(labels), args.num_conditions
        )
        propagated = utils.propagate_hypergraph(
            incidence, weak_labels, alpha=args.alpha, max_iter=args.steps
        )[args.steps]
        results["Weak labels"].append(utils.evaluate(
            weak_labels, test.index, answers, multiclass,
            len(labels), args.num_conditions,
        ))
        results["TabHGR"].append(utils.evaluate(
            propagated, test.index, answers, multiclass,
            len(labels), args.num_conditions,
        ))

    for name, result in results.items():
        values = np.asarray(result)
        print(
            f"{name:<12} mean={values.mean():.4f} std={values.std():.4f} "
            f"min={values.min():.4f} max={values.max():.4f}"
        )


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    features, test, answers, labels, is_categorical = load_dataset(
        args.dataset, args.seed
    )
    num_groups = args.num_groups or min(max(len(features) // 10, 200), 1000)

    if not args.evaluate_only:
        feature_description = utils.get_feature_desc(
            features, is_categorical, f"data/{args.dataset}-metadata.json"
        )
        generate(args, features, labels, feature_description, num_groups)

    evaluate(args, features, test, answers, labels, num_groups)


if __name__ == "__main__":
    main()
