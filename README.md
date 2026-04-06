# GIDS Observer Framework

This project turns the operational math in **Part 2**, **Part 3**, and **Part 4** of *God’s Infinite Dimensional Space* into a small Python framework and a set of executed Jupyter notebooks. This framework isn't proof that GIDS is real, it is only a live check to ensure that the math in the paper works as expected and that people can modify this for their own projects.  

Note: The toy example data here is sparse because I do not want your computer to explode. A real implementation of this, besides figuring out your own embedding framework, would be computationally expensive and probably couldn't run on anything but a couple of graphics cards. Also, this is very unoptimized.

This code is trying to make the operational algebra legible enough that another ML team can:
1. define a slow person-side embedding,
2. define a fast task-conditioned state,
3. represent propositions in the same task-space,
4. predict the next task-relevant observer-state,
5. decode visible outcomes from it, and
6. benchmark the whole thing hard enough that the story either works or dies.

The patched equations are baked in:
- training objective for gradient descent  
  `L = main + Σ λ_probe * probe + λ_reg * reg`
- slow embedding refresh  
  `T_hat <- (1 - α) * T_hat + α * T_hat_new`

## Start here

The notebooks are the easiest entry point.

- `notebooks/00_framework_overview.ipynb` — project map and paper-to-code map
- `notebooks/01_part2_embedding_and_memory.ipynb` — salience, memory, contextual lifting, categorical pooling, slow/fast banks
- `notebooks/02_part3_world_model_and_search.ipynb` — world model, corrected objective, proposition search
- `notebooks/03_part4_benchmarking_and_results.ipynb` — temporal benchmark, ablations, IPS, and interpretation
- `notebooks/04_ml_team_adaptation_guide.ipynb` — how another ML team should adapt the framework to real data

## Project layout

```text
gids_observer_framework_project/
├── docs/
├── notebooks/
├── results/
├── src/gids_observer_framework/
│   ├── benchmark.py
│   ├── categorical.py
│   ├── embedding.py
│   ├── math_utils.py
│   ├── memory.py
│   ├── objective.py
│   ├── ope.py
│   ├── references.py
│   ├── state.py
│   ├── toy_data.py
│   └── experiments/
└── tests/
```

## Paper to code map

| Paper area | Main idea | Code |
|---|---|---|
| Part 2 | salience slice, predictive state, memory as weighted traces | `embedding.py`, `memory.py` |
| Part 2 | contextual lifting and categorical pooling | `categorical.py` |
| Part 2 | first operational slow embedding estimate | `embedding.py`, `toy_data.py` |
| Part 3 | world model and proposition search | `state.py` |
| Part 3 / 4 | corrected training objective | `objective.py` |
| Part 4 | temporal split, benchmark, ablations | `benchmark.py`, `experiments/run_benchmark.py` |
| Part 4 | off-policy evaluation | `ope.py` |

A fuller paper-to-code table lives in `docs/paper_to_code_map.md`.

## Running the project

```bash
cd gids_observer_framework_project
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m gids_observer_framework.experiments.run_all
jupyter lab
```

## Results snapshot

### 1) Significant equation checks
The project runs **12** toy checks over the major operational equations. In this build, **12** checks passed.

### 2) Corrected objective search
Among candidate loss forms, the clean gradient-descent version

```text
L = main + λ_probe * probe + λ_reg * reg
```

finished with mean validation BCE about **0.163 / 0.171** for main/probe losses.  
The paper’s old sign pattern

```text
L = main - λ_probe * probe - λ_reg * reg
```

finished around **0.428 / 13.475**, which is exactly the wrong direction for probe learning.

### 3) Slow-update search
The best slow refresh rule was the EMA-to-new-target form:

```text
T_hat <- (1 - α) * T_hat + α * T_hat_new
```

Mean tracking RMSE: **0.139**  
Literal-current-delta version RMSE: **0.244**

Under shock, the EMA rule had deviation **0.456**, while immediate overwrite jumped to **1.987**.

### 4) Benchmark headline
On the synthetic main task, the explicit latent-state family beat the weaker baselines. There is also a higher area under the curve for the full model. 

| Model | LogLoss | AUC |
|---|---:|---:|
| latent_full | 0.630 | 0.694 |
| latent_no_fast | 0.643 | 0.670 |
| latent_no_slow | 0.638 | 0.683 |
| static | 0.660 | 0.651 |
| monolithic_sequence | 0.685 | 0.634 |

Useful deltas:
- latent_full vs static: **0.030** lower log loss
- latent_full vs no-fast: **0.013** lower log loss
- latent_full vs no-slow: **0.008** lower log loss
- latent_full vs monolithic sequence baseline: **0.055** lower log loss

One ablation is worth reading honestly: on this toy regime, **collapsed slow pooling was nearly tied with the full slow representation**. That is not a bug in the project. Treat architectural claims as hypotheses, run the ablation, and keep only what produces signal.

### 5) IPS sanity check
The IPS demo came out at **0.612** against a true policy value of **0.609**.  
The naive matched mean was **0.577**.

That is the exact practical point from Part 4: without logged propensities or randomization, you are doing ranking, not clean offline policy evaluation.

## What the toy results mean

- The **slow/fast decomposition can buy real signal** when outcomes depend on both durable traits and recent interaction state.
- The **objective sign really matters**. If the probes and regularizer enter with the wrong sign under gradient descent, the model learns against itself.
- The **EMA slow update is the right compromise** for durable embeddings: responsive enough to move, slow enough not to thrash.
- The **categorical machinery is operationally useful** because many real traces arrive as repeated discrete markers rather than dense numeric vectors.
- The **ablation discipline is part of the framework**. If a design choice does not help on your data, drop it.

## Where this is directly usable

This project is shaped so a real ML team can swap the toy generator for production data in domains like:
- GTM / sales interaction modeling
- support and escalation routing
- retention / lifecycle intervention modeling
- recruiting outreach optimization
- care-management or outreach sequencing in regulated settings

The right way to read those examples is not “the toy benchmark proved all of this.”  
The right way to read them is: **the package gives you a clean scaffold for testing the same state logic on your own domain without having to rewrite the formalism from scratch.**

## Read next

- `docs/paper_to_code_map.md`
- `docs/results_and_applicability.md`
- `notebooks/03_part4_benchmarking_and_results.ipynb`
