# TabHGR
Code and datasets of **Zero-Shot Tabular Learning via Hypergraph Reasoning with LLMs** ([EMNLP 2026](https://2026.emnlp.org/)).

**Paper:** [arXiv](https://arxiv.org/abs/2610.23923)

## Tabular Datasets
Experiments can be run on nine tabular datasets: `adult`, `bank`, `blood`, `car`, `credit-g`, `diabetes`, `heart`, `communities`, and `myocardial`.

## Installation
Install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```ㅍ

## Generating Weak Labels and Groups (Optional)
The generated weak labels and groups are already included. To generate them again:

```bash
export OPENAI_API_KEY="your-api-key"
python generate_weak_labels.py --dataset diabetes
python generate_groups.py --dataset diabetes
```

## Running TabHGR and TabHGR+
Run TabHGR and TabHGR+ on a dataset:

```bash
python main.py --dataset diabetes
```

To run specific trials:

```bash
python main.py --dataset diabetes --trials 0
python main.py --dataset diabetes --trials 1,2,3,4
```

If memory is limited, run one trial at a time:

```bash
OPENBLAS_NUM_THREADS=1 python main.py --dataset adult --trials 0
```

To view all available options:

```bash
python main.py --help
```
