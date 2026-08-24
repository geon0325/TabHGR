"""Run the TabHGR and TabHGR+ zero-shot evaluation."""

import argparse
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import trange

import utils


SEED = 42
GROUP_BATCH_SIZE = 20


@dataclass(frozen=True)
class ExperimentData:
    num_rows: int
    test_indices: np.ndarray
    targets: np.ndarray
    num_labels: int

    @property
    def multiclass(self):
        return self.num_labels > 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="diabetes", choices=utils.TASK_DICT)
    parser.add_argument("--pseudo-ratio", "--pseudo_ratio", type=float, default=0.5)
    parser.add_argument("--steps", "--iter", dest="steps", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--num-conditions", "--num_conditions", type=int, default=20)
    parser.add_argument("--num-examples", "--num_examples", type=int, default=5)
    parser.add_argument("--num-trials", "--num_trials", type=int, default=5)
    parser.add_argument(
        "--trials",
        help="comma-separated trial indices (default: 0..num-trials-1)",
    )
    parser.add_argument("--llm-model", "--llm_model", default="gpt-4o")
    return parser.parse_args()


def parse_trials(value, num_trials):
    if value is None:
        return list(range(num_trials))
    return [int(trial) for trial in value.split(",")]


def load_data(dataset):
    dataframe, _, test, _, _, target, labels, _ = utils.get_dataset(
        dataset, SEED, shot=0
    )
    targets = np.array([labels.index(value) for value in dataframe[target]])
    return ExperimentData(
        num_rows=len(dataframe),
        test_indices=test.index.to_numpy(),
        targets=targets,
        num_labels=len(labels),
    )


def num_groups_for(num_rows):
    return min(max(num_rows // 10, 200), 1000)


def group_path(args, num_groups):
    filename = (
        f"{args.dataset}_{GROUP_BATCH_SIZE}_{num_groups}_"
        f"{args.num_examples}_indices.pkl"
    )
    return Path("groups") / args.llm_model / filename


def condition_path(args, trial):
    filename = (
        f"{args.dataset}_{args.num_conditions}_{SEED}_"
        f"{args.num_examples}_condition2index.pkl"
    )
    return Path("conditions/0_shot") / f"{trial}_trial" / args.llm_model / filename


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_rule_features(args, data, trial):
    condition_indices = load_pickle(condition_path(args, trial))
    return utils.generate_feature(
        condition_indices,
        data.num_rows,
        data.num_labels,
        args.num_conditions,
    )


def propagate(incidence, features, args, weights=None):
    return utils.propagate_hypergraph(
        incidence,
        features,
        alpha=args.alpha,
        max_iter=args.steps,
        w=weights,
    )[args.steps]


def make_pseudo_labels(propagated, data, args):
    compatibility = propagated.reshape(
        -1, data.num_labels, args.num_conditions
    ).sum(axis=2)
    probabilities = compatibility / compatibility.sum(axis=1, keepdims=True)
    sorted_scores = np.sort(compatibility, axis=1)
    confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
    nodes = np.argsort(-confidence)[:int(len(confidence) * args.pseudo_ratio)]

    nodes_by_label = defaultdict(list)
    maximum_per_label = math.ceil(len(nodes) / data.num_labels)
    for node in nodes:
        for label in np.argsort(-probabilities[node]):
            if len(nodes_by_label[label]) < maximum_per_label:
                nodes_by_label[label].append(node)
                break

    labels = np.full(data.num_rows, -1)
    for label in range(data.num_labels):
        labels[nodes_by_label[label]] = label
    return nodes, labels


def individual_rule_aurocs(features, nodes, labels, data, args):
    return utils.evaluate_individual_rules(
        features,
        nodes,
        labels,
        data.multiclass,
        data.num_labels,
        args.num_conditions,
    )


def estimate_hyperedge_effectiveness(
    incidence, rule_features, propagated, pseudo_nodes, pseudo_labels, data, args
):
    full_aurocs = individual_rule_aurocs(
        propagated, pseudo_nodes, pseudo_labels, data, args
    )
    removed_aurocs = np.empty((rule_features.shape[1], incidence.shape[1]))

    for edge in trange(incidence.shape[1], desc="leave-one-edge-out"):
        keep_edges = np.arange(incidence.shape[1]) != edge
        propagated_without_edge = propagate(
            incidence[:, keep_edges], rule_features, args
        )
        removed_aurocs[:, edge] = individual_rule_aurocs(
            propagated_without_edge,
            pseudo_nodes,
            pseudo_labels,
            data,
            args,
        )

    return full_aurocs[:, None] - removed_aurocs + 1e-10


def effectiveness_to_weights(effectiveness, tau, num_edges):
    weights = np.exp(effectiveness / tau)
    weights /= weights.sum()
    weights *= num_edges / weights.sum()
    return weights


def adaptive_propagation(incidence, rule_features, effectiveness, args):
    propagated_rules = []
    for rule in range(rule_features.shape[1]):
        weights = effectiveness_to_weights(
            effectiveness[rule], args.tau, incidence.shape[1]
        )
        propagated_rules.append(propagate(
            incidence, rule_features[:, rule:rule + 1], args, weights
        ))
    return np.concatenate(propagated_rules, axis=1)


def evaluate(features, data, args):
    return utils.evaluate(
        features,
        data.test_indices,
        data.targets,
        data.multiclass,
        data.num_labels,
        args.num_conditions,
    )


def run_trial(trial, incidence, data, args):
    rule_features = load_rule_features(args, data, trial)
    propagated = propagate(incidence, rule_features, args)
    pseudo_nodes, pseudo_labels = make_pseudo_labels(propagated, data, args)
    effectiveness = estimate_hyperedge_effectiveness(
        incidence,
        rule_features,
        propagated,
        pseudo_nodes,
        pseudo_labels,
        data,
        args,
    )
    adaptive = adaptive_propagation(
        incidence, rule_features, effectiveness, args
    )
    return np.array([
        evaluate(rule_features, data, args),
        evaluate(propagated, data, args),
        evaluate(adaptive, data, args),
    ])


def print_header(args, incidence):
    print(
        f"dataset={args.dataset} rules/class={args.num_conditions} "
        f"alpha={args.alpha} steps={args.steps} "
        f"q={args.pseudo_ratio:g} tau={args.tau:g}"
    )
    print(f'{"Metric":<10} | {"Weak labels":>11} | {"TabHGR":>8} | {"TabHGR+":>8}')
    print("-" * 49)
    print("Incidence matrix", incidence.shape)


def print_result(label, values):
    print(f"{label:<10} | " + " | ".join(f"{value:8.4f}" for value in values))


def main():
    args = parse_args()
    trials = parse_trials(args.trials, args.num_trials)
    utils.set_seed(SEED)
    data = load_data(args.dataset)
    groups = load_pickle(group_path(args, num_groups_for(data.num_rows)))
    incidence = utils.generate_hypergraph(data.num_rows, groups)

    print_header(args, incidence)
    results = []
    for trial in trials:
        result = run_trial(trial, incidence, data, args)
        results.append(result)
        print_result(f"TRIAL {trial}", result)

    results = np.asarray(results)
    print("-" * 49)
    print_result("MEAN", results.mean(axis=0))
    print_result("STD", results.std(axis=0))
    print()


if __name__ == "__main__":
    main()
