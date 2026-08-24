# TabHGR
Code and datasets of **Zero-Shot Tabular Learning via Hypergraph Reasoning with LLMs**, ([EMNLP 2026](https://2026.emnlp.org/)).

## Tabular Datasets
We provide nine tabular datasets: `adult`, `bank`, `blood`, `car`, `credit-g`, `diabetes`, `heart`, `communities`, and `myocardial`.

## Running TabHGR and TabHGR+
Install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

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
