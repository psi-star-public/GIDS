# Paper to code map

This is the plain engineering index for the paper.

The paper’s prose is ambitious on purpose. The code should be legible on purpose.

## Part 2 → Person-side object, memory, and categorical trace logic

| Paper section | Equation / idea | Notebook | Module |
|---|---|---|---|
| Dimensionality Reduction, Attention, and Relevance | `z_{i,t} = a_{i,t} ⊙ π_τ(T_i, c_{i,t}, x_t)` | `01_part2_embedding_and_memory.ipynb` | `embedding.py` |
| The Notion of State / Observable Predictive State | predictive sufficiency `P(Y|H,T,c,w,x) = P(Y|q,x)` | `01_part2_embedding_and_memory.ipynb` | `experiments/run_toy_equations.py` |
| Memory as a Series of Vectors | `m_{i,t} = Σ ω_{ij,t} μ_{ij}` | `01_part2_embedding_and_memory.ipynb` | `memory.py` |
| Categorical Trace Pooling as an Operational Memory Estimator | contextual lift `\widetilde C = \Xi(C,c)` | `01_part2_embedding_and_memory.ipynb` | `categorical.py` |
| Categorical Trace Pooling as an Operational Memory Estimator | event pooling `u_{i,t}^{(f,s)}` | `01_part2_embedding_and_memory.ipynb` | `categorical.py` |
| Categorical Trace Pooling as an Operational Memory Estimator | slow regime bank `g_i^{slow}` | `01_part2_embedding_and_memory.ipynb` | `categorical.py` |
| Categorical Trace Pooling as an Operational Memory Estimator | fast task-conditioned pool `g_{i,t}^{fast,τ}` | `01_part2_embedding_and_memory.ipynb` | `categorical.py` |
| Deriving the Transcendental Embedding | first-pass slow estimate `\hat T_i^{(0)}` | `01_part2_embedding_and_memory.ipynb` | `embedding.py`, `toy_data.py` |

## Part 3 → World model, corrected objective, proposition search

| Paper section | Equation / idea | Notebook | Module |
|---|---|---|---|
| Towards a Universal State Transition Function | `\hat q_{i,t+1}^{(τ,Δ)} = F_τ(\hat T_i, z_{i,t}, c_{i,t}, w_t, x_t)` | `02_part3_world_model_and_search.ipynb` | `state.py` |
| Towards a Universal State Transition Function | `\hat y = R_0(\hat q)` | `02_part3_world_model_and_search.ipynb` | `state.py` |
| God’s Infinite Dimensional Space: Making All Realities Composable | projection equivalence of propositions | `02_part3_world_model_and_search.ipynb` | `experiments/run_toy_equations.py` |
| Algorithm 3 / Training objective | corrected gradient-descent objective | `02_part3_world_model_and_search.ipynb` | `objective.py` |
| From Forecasting to Proposition Search | `x_t^* ∈ argmax_x score_θ(x|s_t)` | `02_part3_world_model_and_search.ipynb` | `state.py` |

## Part 4 → Benchmarking, temporal splits, OPE, and interpretation

| Paper section | Equation / idea | Notebook | Module |
|---|---|---|---|
| Operational Definition of State | measurable approximation `s_{i,t}^{(τ,Δ)}` | `03_part4_benchmarking_and_results.ipynb` | `toy_data.py` |
| Dataset Construction | synthetic benchmark dataset and feature views | `03_part4_benchmarking_and_results.ipynb` | `toy_data.py` |
| The Benchmark | baseline suite and ablations | `03_part4_benchmarking_and_results.ipynb` | `benchmark.py` |
| Training Objective, Update Loop, and Intervention | corrected loss and EMA slow update | `02_part3_world_model_and_search.ipynb`, `03_part4_benchmarking_and_results.ipynb` | `objective.py`, `state.py`, `experiments/run_candidate_search.py` |
| Off-policy evaluation | IPS | `03_part4_benchmarking_and_results.ipynb` | `ope.py` |
| Temporal Split, Evaluation, and Drift | train/val/test over time | `03_part4_benchmarking_and_results.ipynb` | `benchmark.py` |

## The intended reading order

1. `00_framework_overview.ipynb`
2. `01_part2_embedding_and_memory.ipynb`
3. `02_part3_world_model_and_search.ipynb`
4. `03_part4_benchmarking_and_results.ipynb`
5. `04_ml_team_adaptation_guide.ipynb`

## The intended engineering order

1. Define the task and horizon.
2. Decide what counts as static person/account context.
3. Decide which categorical families, sources, and regimes you want to preserve.
4. Build the slow bank and fast pool.
5. Fit a simple latent-state model and the weaker baselines.
6. Run ablations before you get attached to the architecture.
7. Only move into policy evaluation if propensities or experiments exist.
