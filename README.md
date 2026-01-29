# Zero-Shot Tabular Learning via Hypergraph Reasoning with LLMs

This repository contains the official implementation for **TabHGR+**, a zero-shot tabular learning framework based on hypergraph reasoning using LLMs. 


## Repository Structure
```graphql
.
├── data/                      # Datasets
├── generate_weak_labels.ipynb  # LLM-based weak label generation
├── generate_groups.ipynb       # Hypergraph (group) construction
├── main.py                     # Training & inference for TabHGR+
└── README.md
```

## Datasets
All datasets used in the paper are provided in the [data] directory.

## Weak Label Generation
We use large language models (LLMs) to generate weak labels based on task and attribute descriptions.

Refer to [generate_weak_labels.ipynb]

## Hyeprgraph Generation
We construct a hypergraph where each hyperedge represents a semantic group of related rows in the table.

Refer to [generate_groups.ipynb]

## TabHGR+
TabHGR+ performs hypergraph propagation over the constructed hypergraph with label-adaptive hyperedge weighting.

```bash
python main.py --dataset diabetes
```