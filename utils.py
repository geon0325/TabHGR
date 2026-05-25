# Utility function for getting data & prompting & query
import os
import re
import sys
import random
import importlib
from openai import OpenAI
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    attribute_names = df.columns[:-1].tolist()

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
        total_shot_count = 0
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


def get_feature_desc(df, is_cat, meta_file_name, numbered=False, shuffled=False):     
    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except:
        meta_data = {}

    columns = df.columns.tolist()
                
    feature_name_list = []
    sel_cat_idx = [df.columns.tolist().index(col_name) for col_name in columns]
    is_cat_sel = np.array(is_cat)[sel_cat_idx]
    
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


def query_gpt(text, api_key, max_tokens=30, temperature=0, max_try_num=10, model='gpt-4o-mini'):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":text}],
        temperature = temperature,
        #max_tokens = max_tokens,
        top_p = 1
    )
    result = response.choices[0].message.content
    return result

def parse_conditions(text, label_list, splitter="Label"):
    splitted = text.split(splitter)
    rule_raws = splitted[1:]
    rule_dict, rule_desc = {}, {}
    for rule_raw in rule_raws:
        #label_index = [l in [x.lower() for x in rule_raw.split()] for l in label_list].index(True)
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
    #rule_dict = [rule_dict[group_title] for label_index in sorted(rule_dict.keys())]
    return group_titles, group_conditions

def parse_group_descriptions(text, splitter="Group"):
    splitted = text.split(splitter)
    descriptions = splitted[1:]
    descriptions = [description.split(':')[1].strip(" .'").strip(' []"').strip('\n') for description in descriptions]
    return descriptions

def parse_conditions_single_group(text):
    splitted = text.split('\n')
    splitted = [s.split('-')[1].strip() for s in splitted]
    return splitted

def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str    

def get_prompt(template_file, fill_in_dict):
    with open(template_file, "r") as f:
        prompt_type_str = f.read()
    
    template = fill_in_templates(fill_in_dict, prompt_type_str)
    
    return template


def run_function(function_str, df):
    match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', function_str)
    fct_name = match.group(1)

    seed = random.randint(1, 10000000)

    while os.path.exists(f'temp_{seed}.py'):
        seed = random.randint(1, 10000000)

    with open(f'temp_{seed}.py', 'w') as f:
        f.write(function_str)

    patience = 0
    while True:
        try:
            spec = importlib.util.spec_from_file_location("temp_module", f'temp_{seed}.py')
            temp_module = importlib.util.module_from_spec(spec)
            sys.modules["temp_module"] = temp_module
            spec.loader.exec_module(temp_module)
            break
        except:
            patience += 1

        if patience == 10:
            break

    if patience == 10:
        os.remove(f'temp_{seed}.py')
        return "Error 1"

    try:
        fct = getattr(temp_module, fct_name)
    except:
        spec = importlib.util.spec_from_file_location("temp_module", f'temp_{seed}.py')
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module"] = temp_module
        spec.loader.exec_module(temp_module)

    try:
        index = fct(df)
    except:
        os.remove(f'temp_{seed}.py')
        return "Error 2"

    if isinstance(index, type(df)):
        index = index.index
    
    try:
        index = index.tolist()
    except:
        pass

    os.remove(f'temp_{seed}.py')
    
    return index

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
    H = np.zeros((num_sample, num_hyperedge))
    
    # Fill incidence matrix H
    for cnt_hyperedge, node_indices in enumerate(group2index):
        H[node_indices, cnt_hyperedge] = 1

    return H

def edge_node_edge_propagation(H, X, alpha=0.9, max_iter=50, w=None):
    """
    H: (N x E) incidence matrix
    X: (N x d) node feature matrix
    w: (E,) or (E, 1) hyperedge weight vector (optional)
    """
    N, E = H.shape
    node_degree = H.sum(axis=1, keepdims=True) + 1e-6
    edge_degree = H.sum(axis=0, keepdims=True) + 1e-6

    # handle weight vector
    if w is None:
        w = np.ones((E, 1))
    else:
        w = np.asarray(w).reshape(E, 1)

    F = X.copy()
    F_list = [F]

    for _ in range(max_iter):
        # Step 1: nodes → hyperedges (aggregation)
        edge_feat = (H.T @ (F / node_degree)) * w  # [E x d]

        # Step 2: hyperedges → nodes (redistribution)
        node_feat = H @ ((edge_feat * w) / edge_degree.T)  # [N x d]

        # Step 3: combine with initial features
        F = alpha * node_feat + (1 - alpha) * X
        F_list.append(F)

    return F_list

def propagate_hypergraph(H, X, alpha=0.9, max_iter=10, w=None):
    H_list = edge_node_edge_propagation(H, X, alpha, max_iter, w)
    return H_list
    
def compute_auroc(answers, pred_probs, multiclass=False):   
    if multiclass == False:
        result_auc = roc_auc_score(answers, pred_probs[:, 1])
    else:
        result_auc = roc_auc_score(answers, pred_probs, multi_class='ovr', average='macro')        
    return result_auc


def evaluate(Z, x_index, y, multiclass, num_label, num_condition):
    # --- Normalize for prediction ---
    Y = Z.reshape(-1, num_label, num_condition).sum(axis=2) + 1e-10
    Y = Y / Y.sum(axis=1, keepdims=True)

    # --- Evaluate ---
    auroc = compute_auroc(y[x_index], Y[x_index], multiclass)

    return auroc
