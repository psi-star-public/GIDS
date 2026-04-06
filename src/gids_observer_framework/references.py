from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass(frozen=True)
class PaperReference:
    key: str
    part: str
    section: str
    equation: str
    implementation: str
    why_it_matters: str


PAPER_REFERENCES: List[PaperReference] = [
    PaperReference(
        key="salience_slice",
        part="Part 2",
        section="Dimensionality Reduction, Attention, and Relevance",
        equation=r"z_{i,t} = a_{i,t} \odot \pi_\tau(T_i, c_{i,t}, x_t)",
        implementation="embedding.salience_slice",
        why_it_matters="The transition often depends on a weighted slice of the person-space, not the whole thing at once.",
    ),
    PaperReference(
        key="predictive_state",
        part="Part 2 / Part 3 / Part 4",
        section="The Notion of State / Towards a Universal State Transition Function / Operational Definition of State",
        equation=r"P(Y\mid H, T, c, w, x) = P(Y\mid q, x)",
        implementation="experiments.run_toy_equations.predictive_sufficiency_demo",
        why_it_matters="This is the honest formal target: task-conditioned predictive state, not the whole ineffable interior.",
    ),
    PaperReference(
        key="memory_field",
        part="Part 2",
        section="Memory as a Series of Vectors",
        equation=r"m_{i,t} = \sum_j \omega_{ij,t} \mu_{ij}",
        implementation="memory.memory_field",
        why_it_matters="Repeated traces become a weighted field instead of a vague story about memory.",
    ),
    PaperReference(
        key="contextual_lift",
        part="Part 2",
        section="Categorical Trace Pooling as an Operational Memory Estimator",
        equation=r"\widetilde{C}_{i,t}^{(f,s)} = \Xi(C_{i,t}^{(f,s)}, c_{i,t})",
        implementation="categorical.contextual_lift",
        why_it_matters="Surface contradictions often disappear once you type the asymmetry carried by the regime.",
    ),
    PaperReference(
        key="categorical_pooling",
        part="Part 2",
        section="Categorical Trace Pooling as an Operational Memory Estimator",
        equation=r"u_{i,t}^{(f,s)} = \frac{1}{|\widetilde{C}|} \sum_{c \in \widetilde{C}} E_{f,s}(c)",
        implementation="categorical.build_event_categorical_embedding",
        why_it_matters="Sparse categorical traces become fixed-width vectors without pretending missing slots do not exist.",
    ),
    PaperReference(
        key="slow_bank",
        part="Part 2",
        section="Categorical Trace Pooling as an Operational Memory Estimator",
        equation=r"g_{i,\rho}^{\mathrm{slow}} = \text{weighted regime average of } e_{i,r}^{\mathrm{cat}}",
        implementation="categorical.build_slow_bank",
        why_it_matters="Durable, regime-aware person structure should not be collapsed into one global mush.",
    ),
    PaperReference(
        key="fast_pool",
        part="Part 2",
        section="Categorical Trace Pooling as an Operational Memory Estimator",
        equation=r"g_{i,t}^{\mathrm{fast},\tau} = \sum_{r \le t} \alpha_{i,r,t}^{(\tau)} e_{i,r}^{\mathrm{cat}}",
        implementation="categorical.build_fast_pool",
        why_it_matters="Recent decisive traces and accumulated weak exposure both matter, but not equally.",
    ),
    PaperReference(
        key="slow_embedding",
        part="Part 2",
        section="Deriving the Transcendental Embedding",
        equation=r"\hat T_i^{(0)} = E(p_i, b_i, \ell_i, r_i, h_i, g_i^{\mathrm{slow}})",
        implementation="embedding.estimate_slow_embedding",
        why_it_matters="This is the first operational estimate of the person-side embedding.",
    ),
    PaperReference(
        key="world_model",
        part="Part 3 / Part 4",
        section="Creating the World Model / The Proposed Latent-State Model",
        equation=r"\hat q_{i,t+1}^{(\tau,\Delta)} = F_\tau(\hat T_i, z_{i,t}, c_{i,t}, w_t, x_t)",
        implementation="state.world_model_step",
        why_it_matters="A world model here is a simulator of predictive-state transitions under propositions.",
    ),
    PaperReference(
        key="readout",
        part="Part 3 / Part 4",
        section="Towards a Universal State Transition Function / The Proposed Latent-State Model",
        equation=r"\hat y_{i,t+\Delta}^{(\tau)} = R_0(\hat q_{i,t+1}^{(\tau,\Delta)})",
        implementation="state.readout_probability",
        why_it_matters="Observed outcomes are visible residues of the transition, not the state itself.",
    ),
    PaperReference(
        key="training_objective",
        part="Part 3 / Part 4",
        section="Algorithm 3 / Training Objective, Update Loop, and Intervention",
        equation=r"\mathcal L_\tau = \mathcal L_{\mathrm{main}} + \sum_m \lambda_m \mathcal L_{\mathrm{probe},m} + \lambda_{\mathrm{reg}}\Omega(\theta)",
        implementation="objective.total_loss",
        why_it_matters="With gradient descent, the probe and regularization terms have to help rather than fight the optimization.",
    ),
    PaperReference(
        key="slow_update",
        part="Part 4",
        section="Training Objective, Update Loop, and Intervention",
        equation=r"\hat T_i \leftarrow (1-\alpha)\hat T_i + \alpha\hat T_i^{\mathrm{new}}",
        implementation="state.ema_slow_update",
        why_it_matters="The durable embedding should move slowly toward refreshed evidence, not get jerked around by one event.",
    ),
    PaperReference(
        key="proposition_search",
        part="Part 3 / Part 4",
        section="From Forecasting to Proposition Search / Observational Ranking",
        equation=r"x_t^\star \in \arg\max_{x \in \mathcal X_{i,t}^{\mathrm{adm}}} \operatorname{score}_\theta(x \mid s_{i,t})",
        implementation="state.best_proposition",
        why_it_matters="The point is not only to predict what happened, but to rank admissible next propositions.",
    ),
    PaperReference(
        key="ips",
        part="Part 4",
        section="Off-policy evaluation",
        equation=r"\hat V_{\mathrm{IPS}}(\pi) = \frac{1}{N}\sum_t \frac{\mathbb 1\{x_t = \pi(s_t)\}}{e_t} r_t",
        implementation="ope.ips_value",
        why_it_matters="Ranking is not causality; logged propensities are the bridge to offline policy evaluation.",
    ),
    PaperReference(
        key="temporal_split",
        part="Part 4",
        section="Temporal Split, Evaluation, and Drift",
        equation=r"\mathcal D_{\mathrm{train}}^{(1:T_1)}, \mathcal D_{\mathrm{val}}^{(T_1:T_2)}, \mathcal D_{\mathrm{test}}^{(T_2:T_3)}",
        implementation="benchmark.temporal_split",
        why_it_matters="Random row splits cheat. Temporal splits make the benchmark answerable to time.",
    ),
]


def PAPER_REFERENCE_TABLE():
    return [asdict(item) for item in PAPER_REFERENCES]
