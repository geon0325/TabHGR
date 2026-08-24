### Zero-Shot Tabular Learning via Hypergraph Reasoning with LLMs

Geon Lee, Kyungho Kim, Dongwon Choi, Juyeon Kim, Shinhwan Kang, and Kijung Shin

[**EMNLP 2026**](https://2026.emnlp.org/): Conference on Empirical Methods in Natural Language Processing 2026

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running

To run TabHGR and TabHGR+ on the Diabetes dataset:

```bash
python main.py --dataset diabetes
```

Other datasets can be evaluated in the same way:

```bash
python main.py --dataset heart
```

Supported datasets: `adult`, `bank`, `blood`, `car`, `credit-g`, `diabetes`, `heart`, `communities`, and `myocardial`.

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
