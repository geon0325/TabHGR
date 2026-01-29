import os
import math
import utils
import random
import argparse
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='diabetes')
parser.add_argument('--pseudo_ratio', type=float, default=0.5)
args = parser.parse_args()

_DATA = args.dataset
_NUM_CONDITION = 20
_SEED = 42
_SHOT = 0
_NUM_EXAMPLE = 5
_NUM_TRIAL = 5
llm_model = 'gpt-4o'

print(args.dataset, args.pseudo_ratio)

utils.set_seed(_SEED)
df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(_DATA, _SEED, _SHOT)
multiclass = True if len(label_list) > 2 else False
y_test = [label_list.index(l) for l in y_test]

X_all = df.drop(target_attr, axis=1)
y_all = np.array([label_list.index(l) for l in df[target_attr]])

if _DATA in ['communities', 'myocardial']:
    with open(f'data/{_DATA}_selected_column_indices.pkl', 'rb') as f:
        selected_column_indices = pkl.load(f)

    X_all = X_all.iloc[:,selected_column_indices]
    X_test = X_test.iloc[:,selected_column_indices]

    if X_train is not None:
        X_train = X_train.iloc[:,selected_column_indices]

_NUM_GROUP_TOTAL = min(max(X_all.shape[0] // 10, 200), 1000)
_NUM_GROUP_BATCH = 20

pseudo_nodes = []
pseudo_labels = []

with open(f'groups/{llm_model}/{_DATA}_{_NUM_GROUP_BATCH}_{_NUM_GROUP_TOTAL}_{_NUM_EXAMPLE}_indices.pkl', 'rb') as f:
    groups = pkl.load(f)
H = utils.generate_hypergraph(X_all.shape[0], groups)

for TRIAL in range(5):    
    ### Load Features (Labels/Conditions) ###
    save_dir = f'conditions/0_shot/{TRIAL}_trial/{llm_model}'

    with open(f'{save_dir}/{_DATA}_{_NUM_CONDITION}_{_SEED}_{_NUM_EXAMPLE}_condition2index.pkl', 'rb') as f:
        condition2index = pkl.load(f)
    
    X = utils.generate_feature(condition2index, X_all.shape[0], len(label_list), _NUM_CONDITION)
    Z = utils.propagate_hypergraph(H, X, alpha=0.9, max_iter=2)[2]
    C = np.sum(Z.reshape(-1, len(label_list), _NUM_CONDITION), axis=2)

    C_prob = C / np.sum(C, axis=1)[:,None]
    sorted_C = np.sort(C, axis=1)
    conf = sorted_C[:, -1] - sorted_C[:, -2]

    nodes =  np.argsort(-conf)[:int(len(conf) * args.pseudo_ratio)]
    labels = np.full(len(C), -1)

    label2nodes = defaultdict(list)
    max_num_label = math.ceil(len(nodes) / len(label_list))
    for node in nodes:
        node_label = np.argsort(-C_prob[node])
        for i in range(len(label_list)):
            if len(label2nodes[node_label[i]]) < max_num_label:
                label2nodes[node_label[i]].append(node)
                break
    labels = np.full(len(C), -1)
    for l in range(len(label_list)):
        labels[label2nodes[l]] = l
    
    pseudo_nodes.append(nodes)
    pseudo_labels.append(labels)
    

### Load Groups ###
with open(f'groups/{llm_model}/{_DATA}_{_NUM_GROUP_BATCH}_{_NUM_GROUP_TOTAL}_{_NUM_EXAMPLE}_groups.pkl', 'rb') as f:
    group_conditions, group_titles = pkl.load(f)

with open(f'groups/{llm_model}/{_DATA}_{_NUM_GROUP_BATCH}_{_NUM_GROUP_TOTAL}_{_NUM_EXAMPLE}_indices.pkl', 'rb') as f:
    groups = pkl.load(f)

# Fixed settings
H = utils.generate_hypergraph(X_all.shape[0], groups)
y_test = y_all[X_test.index]

results_full_pseudo = {}
results_pseudo = {}

for TRIAL in range(5):
    results_full_pseudo[TRIAL] = []
    results_pseudo[TRIAL] = defaultdict(list)
    
    ### Load Features (Labels/Conditions) ###
    save_dir = f'conditions/0_shot/{TRIAL}_trial/{llm_model}'

    with open(f'{save_dir}/{_DATA}_{_NUM_CONDITION}_{_SEED}_{_NUM_EXAMPLE}_condition2index.pkl', 'rb') as f:
        condition2index = pkl.load(f)
    
    X = utils.generate_feature(condition2index, X_all.shape[0], len(label_list), _NUM_CONDITION)
    Z = utils.propagate_hypergraph(H, X, alpha=0.9, max_iter=2)[2]

    for c_idx in range(_NUM_CONDITION * len(label_list)):
        _Z = np.zeros(Z.shape)
        _Z[:,c_idx:c_idx+1] = Z[:,c_idx:c_idx+1]
        auroc = utils.evaluate(_Z, pseudo_nodes[TRIAL], pseudo_labels[TRIAL], multiclass, len(label_list), _NUM_CONDITION)
        results_full_pseudo[TRIAL].append(auroc)

    for i in range(len(groups)):
        _groups = groups[:i] + groups[i+1:]
        H_i = utils.generate_hypergraph(X_all.shape[0], _groups)
        Z = utils.propagate_hypergraph(H_i, X, alpha=0.9, max_iter=2)[2]

        for c_idx in range(_NUM_CONDITION * len(label_list)):
            _Z = np.zeros(Z.shape)
            _Z[:,c_idx:c_idx+1] = Z[:,c_idx:c_idx+1]
            auroc = utils.evaluate(_Z, pseudo_nodes[TRIAL], pseudo_labels[TRIAL], multiclass, len(label_list), _NUM_CONDITION)
            results_pseudo[TRIAL][c_idx].append(auroc)


group_scores_pseudo = {}

for TRIAL in range(5):
    group_scores_pseudo[TRIAL] = defaultdict(list)
    for c_idx in range(_NUM_CONDITION * len(label_list)):
        score_original = results_full_pseudo[TRIAL][c_idx]
        for i in range(len(groups)):    
            score_updated = results_pseudo[TRIAL][c_idx][i]
            group_scores_pseudo[TRIAL][c_idx].append(score_original - score_updated)
        group_scores_pseudo[TRIAL][c_idx] = np.array(group_scores_pseudo[TRIAL][c_idx]) + 1e-10


tau_list = [0.01]

print(f'{"Metric":<10} | {"Raw":>8} | {"Orig":>8} | ' +
      ' | '.join([f"τ={tau:<6}" for tau in tau_list]))
print('-' * (24 + 12 * (len(tau_list) + 2)))

### Load Groups ###
with open(f'groups/{llm_model}/{_DATA}_{_NUM_GROUP_BATCH}_{_NUM_GROUP_TOTAL}_{_NUM_EXAMPLE}_indices.pkl', 'rb') as f:
    groups = pkl.load(f)

H = utils.generate_hypergraph(X_all.shape[0], groups)

all_results = []  # store per-trial results

for TRIAL in range(5):    
    
    ### Load Features (Labels/Conditions) ###
    save_dir = f'conditions/0_shot/{TRIAL}_trial/{llm_model}'
    with open(f'{save_dir}/{_DATA}_{_NUM_CONDITION}_{_SEED}_{_NUM_EXAMPLE}_condition2index.pkl', 'rb') as f:
        condition2index = pkl.load(f)
    
    X = utils.generate_feature(condition2index, X_all.shape[0], len(label_list), _NUM_CONDITION)
    auc_raw = utils.evaluate(X, X_test.index, y_all, multiclass, len(label_list), _NUM_CONDITION)
    
    Z = utils.propagate_hypergraph(H, X, alpha=0.9, max_iter=2)[2]
    auc_orig = utils.evaluate(Z, X_test.index, y_all, multiclass, len(label_list), _NUM_CONDITION)

    Z_minmax_list, Z_sigmoid_list, Z_softmax_list = [], [], defaultdict(list)
    for c_idx in range(_NUM_CONDITION * len(label_list)):
        gsp = np.array(group_scores_pseudo[TRIAL][c_idx])

        norm = True
        for tau in tau_list:
            w_tau = np.exp(gsp / tau)
            w_tau /= np.sum(w_tau)
            w_tau *= len(groups) / np.sum(w_tau)
            Z_tau = utils.propagate_hypergraph(H, X[:, c_idx:c_idx + 1], alpha=0.9, max_iter=2, w=w_tau)[2]
            Z_softmax_list[tau].append(Z_tau)

    # --- Evaluate ---
    Z_minmax = np.concatenate(Z_minmax_list, axis=1)
    auc_minmax = utils.evaluate(Z_minmax, X_test.index, y_all, multiclass, len(label_list), _NUM_CONDITION)

    Z_sigmoid = np.concatenate(Z_sigmoid_list, axis=1)
    auc_sigmoid = utils.evaluate(Z_sigmoid, X_test.index, y_all, multiclass, len(label_list), _NUM_CONDITION)

    auc_softmax_list = []
    for tau in tau_list:
        Z = np.concatenate(Z_softmax_list[tau], axis=1)
        auc = utils.evaluate(Z, X_test.index, y_all, multiclass, len(label_list), _NUM_CONDITION)
        auc_softmax_list.append(auc)
        
    all_results.append([auc_raw, auc_orig, auc_minmax, auc_sigmoid] + auc_softmax_list)

    # Per-trial print
    print(f'{f"TRIAL {TRIAL}":<10} | {auc_raw:8.4f} | {auc_orig:8.4f} | {auc_minmax:8.4f} | {auc_sigmoid:8.4f} |' +
          ' | '.join([f"{auc:8.4f}" for auc in auc_softmax_list]))

# --- Summary ---
all_results = np.array(all_results)
mean_vals = np.mean(all_results, axis=0)
std_vals = np.std(all_results, axis=0)


print('-' * (24 + 12 * (len(tau_list) + 2)))
print(f'{"MEAN":<10} | ' + ' | '.join([f"{m:8.4f}" for m in mean_vals]))
print(f'{"STD":<9}  | ' + ' | '.join([f"{s:8.4f}" for s in std_vals]))
print()
