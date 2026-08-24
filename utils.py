"""Dataset, prompting, hypergraph, and evaluation utilities."""

import re
import sys
import random
import importlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.sparse import csr_matrix

TASK_DICT = {
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'car': "How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?"
}



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    
def get_dataset(data_name, seed, shot=0):
    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [True if (dt == np.dtype('O') or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()][:-1]
    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    if shot > 0:    
        assert(shot <= 128) # We only consider the low-shot regimes here
        X_new_train = X_train.copy()
        X_new_train[default_target_attribute] = y_train
        sampled_list = []
        remainder = shot % len(np.unique(y_train))
        for _, grouped in X_new_train.groupby(default_target_attribute):
            sample_num = shot // len(np.unique(y_train))
            if remainder > 0:
                sample_num += 1
                remainder -= 1
            grouped = grouped.sample(sample_num, random_state=seed)
            sampled_list.append(grouped)
        X_balanced = pd.concat(sampled_list)
        X_train = X_balanced.drop([default_target_attribute], axis=1)
        y_train = X_balanced[default_target_attribute].to_numpy()
    else:
        X_train = None
        y_train = None

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def get_feature_desc(df, is_cat, meta_file_name, numbered=False):
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        meta_data = {}

    columns = df.columns.tolist()
    feature_name_list = []
    is_cat_sel = np.asarray(is_cat)
    
    for cidx, cname in enumerate(columns):
        if is_cat_sel[cidx] == True:
            clist = df[cname].unique().tolist()
            if len(clist) > 20:
                clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
            else:
                clist_str = ", ".join(clist)
            desc = meta_data[cname] if cname in meta_data.keys() else ""
            if numbered:
                feature_name_list.append(f"{cidx+1}. {cname}: {desc} (categorical variable with categories [{clist_str}])")
            else:
                feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
        else:
            desc = meta_data[cname] if cname in meta_data.keys() else ""
            if numbered:
                feature_name_list.append(f"{cidx+1}. {cname}: {desc} (numerical variable)")
            else:
                feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

    feature_desc = "\n".join(feature_name_list)

    return feature_desc


def query_gpt(text, api_key, temperature=0, model='gpt-4o-mini'):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":text}],
        temperature=temperature,
        top_p=1,
    )
    result = response.choices[0].message.content
    return result

def parse_conditions(text, label_list, splitter="Label"):
    splitted = text.split(splitter)
    rule_raws = splitted[1:]
    rule_dict = {}
    for rule_raw in rule_raws:
        label_index = label_list.index(rule_raw.split('\n')[0].split(':')[1].strip(" .'").strip(' []"').lower())
        rule_parsed = []
        for txt in rule_raw.strip().split("\n")[1:]:
            if len(txt) < 2:
                break
            if re.match(r"^(-|\d+\.)", txt.strip()):
                txt_ref = " ".join(txt.strip().split(" ")[1:])
            else:
                txt_ref = " ".join(txt.strip().split(" ")[:])
            rule_parsed.append(txt_ref)
        rule_dict[label_index] = rule_parsed
    conditions = [[r.split(';')[0] for r in rule_dict[label_index]] for label_index in range(len(rule_dict))]
    descriptions = [
        [
            r.split(';')[1].strip() if ';' in r else r.strip()
            for r in rule_dict[label_index]
        ]
        for label_index in range(len(rule_dict))
    ]
    return conditions, descriptions
    
def parse_groups(text, splitter="Group"):
    splitted = text.split(splitter)
    rule_raws = splitted[1:]
    rule_dict = {}
    for rule_raw in rule_raws:
        group_title = rule_raw.split('\n')[0].split(':')[1].strip(" .'").strip(' []"')
        rule_parsed = []
        for txt in rule_raw.strip().split("\n")[1:]:
            if len(txt) < 2:
                break
            rule_parsed.append(txt.lstrip('- ').strip())
        rule_dict[group_title] = rule_parsed
    group_titles = list(rule_dict.keys())
    group_conditions = [rule_dict[group_title] for group_title in group_titles]
    return group_titles, group_conditions

def get_prompt(template_file, fill_in_dict):
    with open(template_file, "r") as f:
        template = f.read()
    for key, value in fill_in_dict.items():
        template = template.replace(key, value)
    return template


def run_function(function_str, df):
    match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', function_str)
    if match is None:
        return "Error 1"
    function_name = match.group(1)
    module_name = f"generated_condition_{random.randint(1, 10_000_000)}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as file:
        file.write(function_str)
        module_path = Path(file.name)

    try:
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            function = getattr(module, function_name)
        except Exception:
            return "Error 1"

        try:
            indices = function(df)
        except Exception:
            return "Error 2"

        if isinstance(indices, type(df)):
            indices = indices.index
        return indices.tolist() if hasattr(indices, "tolist") else indices
    finally:
        sys.modules.pop(module_name, None)
        module_path.unlink(missing_ok=True)

def generate_feature(condition2index, num_sample, num_label, num_condition):
    # --- Initialize condition matrix ---
    X = np.zeros((num_sample, num_condition * num_label))
    
    # --- Populate condition matrix ---
    for label_idx, indices in condition2index.items():
        for cond_offset, row_idx in enumerate(indices):
            if not isinstance(row_idx, list):
                continue
            col_idx = num_condition * label_idx + cond_offset
            X[row_idx, col_idx] = 1

    return X

def generate_hypergraph(num_sample, group2index):
    num_hyperedge = len(group2index)

    # Large datasets such as Adult and Bank would otherwise allocate hundreds
    # of MB and repeatedly multiply mostly-zero dense incidence matrices.
    if num_sample * num_hyperedge >= 10_000_000:
        rows = np.concatenate([
            np.asarray(indices, dtype=np.int64) for indices in group2index
        ])
        cols = np.concatenate([
            np.full(len(indices), edge, dtype=np.int64)
            for edge, indices in enumerate(group2index)
        ])
        return csr_matrix(
            (np.ones(len(rows), dtype=float), (rows, cols)),
            shape=(num_sample, num_hyperedge)
        )

    H = np.zeros((num_sample, num_hyperedge))
    
    # Fill incidence matrix H
    for cnt_hyperedge, node_indices in enumerate(group2index):
        H[node_indices, cnt_hyperedge] = 1

    return H

def propagate_hypergraph(H, X, alpha=0.9, max_iter=10, w=None):
    """Propagate node features through hyperedges for ``max_iter`` steps."""
    _, num_edges = H.shape
    node_degree = np.asarray(H.sum(axis=1)).reshape(-1, 1) + 1e-6
    edge_degree = np.asarray(H.sum(axis=0)).reshape(1, -1) + 1e-6

    if w is None:
        w = np.ones((num_edges, 1))
    else:
        w = np.asarray(w).reshape(num_edges, 1)

    F = X.copy()
    F_list = [F]

    for _ in range(max_iter):
        edge_feat = (H.T @ (F / node_degree)) * w
        node_feat = H @ ((edge_feat * w) / edge_degree.T)
        F = alpha * node_feat + (1 - alpha) * X
        F_list.append(F)

    return F_list


def evaluate(Z, x_index, y, multiclass, num_label, num_condition):
    Y = Z.reshape(-1, num_label, num_condition).sum(axis=2) + 1e-10
    Y = Y / Y.sum(axis=1, keepdims=True)

    if multiclass:
        return roc_auc_score(
            y[x_index], Y[x_index], multi_class='ovr', average='macro'
        )
    return roc_auc_score(y[x_index], Y[x_index, 1])


def evaluate_individual_rules(Z, x_index, y, multiclass, num_label, num_condition):
    """Evaluate every rule column without constructing one-hot score matrices.

    This is mathematically equivalent to calling ``evaluate`` once per column
    after zeroing all other columns, but batches the AUROC calculations. It is
    especially important for multiclass datasets, where the leave-one-edge-out
    effectiveness calculation otherwise makes tens of thousands of sklearn
    calls.
    """
    scores = np.asarray(Z)[x_index]
    answers = np.asarray(y)[x_index]
    num_rules = scores.shape[1]
    rule_classes = np.arange(num_rules) // num_condition

    # Rank-based Mann-Whitney AUROC, vectorized over every rule column.
    ranks = rankdata(scores, axis=0, method='average')
    class_auc = np.empty((num_label, num_rules))
    for label in range(num_label):
        binary_answers = answers == label
        num_positive = binary_answers.sum()
        num_negative = len(binary_answers) - num_positive
        positive_rank_sum = ranks[binary_answers].sum(axis=0)
        class_auc[label] = (
            positive_rank_sum - num_positive * (num_positive + 1) / 2
        ) / (num_positive * num_negative)

    if not multiclass:
        positive_auc = class_auc[1]
        return np.where(rule_classes == 1, positive_auc, 1.0 - positive_auc)

    # For the rule's own class the normalized probability increases with its
    # score; probabilities of all other classes decrease with that score.
    own_auc = class_auc[rule_classes, np.arange(num_rules)]
    other_auc = (num_label - 1) - (class_auc.sum(axis=0) - own_auc)
    return (own_auc + other_auc) / num_label
