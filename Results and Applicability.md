# Results and applicability

This file is the non-notebook version of the results section.

The short version is that the project does what it is supposed to do: it makes the significant equations executable, it stress-tests the two patched equations, and it wraps the whole thing in a benchmark loop that another ML team can replace with its own data.

## 1. Equation checks

The toy battery covered **12** significant equations or formal claims from Parts 2–4.

Everything in the current battery passed. That does **not** prove the metaphysics. It proves the operational math is coherent enough to run.

The main checks include:
- salience slicing
- predictive-state sufficiency on a toy process
- memory as weighted traces
- contextual lifting
- within-event categorical pooling
- slow regime-aware memory
- fast task-conditioned memory
- slow embedding estimation
- projection equivalence of propositions
- world-model rollout
- feature-family contribution
- proposition search

## 2. Patched objective: what won and why

For a gradient-descent training loop, the best paper-aligned equation is still:

```text
L = main + λ_probe * probe + λ_reg * reg
```

Its mean validation BCE came out around:
- main: **0.163**
- probe: **0.171**

The old sign pattern from the draft:

```text
L = main - λ_probe * probe - λ_reg * reg
```

blew the probe term up to about **13.475** on the toy search.

There was one numerically stronger variant, `main + probe - reg`, but it wins by anti-regularizing the weights. That is not faithful to the paper’s own prose about ordinary weight control. So the project keeps the positive-sum objective because it is both effective and semantically aligned.

## 3. Patched slow update: what won and why

The two equivalent good forms were:

```text
T_hat <- (1 - α) * T_hat + α * T_hat_new
```

and

```text
T_hat <- T_hat + α * (T_hat_new - T_hat)
```

They produced the same mean tracking RMSE: **0.139**.

The literal-current-delta interpretation was much worse:
- RMSE: **0.244**
- change-point error: **0.426**

Under shock:
- EMA shock deviation: **0.456**
- overwrite shock deviation: **1.987**

That is the exact reason the code standardizes on the EMA-to-new-target wording. It is the cleanest fit to the paper’s “durable embedding refreshed by accumulating evidence” thesis.

## 4. Benchmark results

### Main target

The explicit latent-state family beat the weaker baseline family on the synthetic main task.

| Model | LogLoss | Brier | PR-AUC | AUC |
|---|---:|---:|---:|---:|
| latent_full | 0.630 | 0.220 | 0.649 | 0.694 |
| latent_no_fast | 0.643 | 0.226 | 0.643 | 0.670 |
| latent_no_slow | 0.638 | 0.224 | 0.652 | 0.683 |
| static | 0.660 | 0.234 | 0.591 | 0.651 |
| monolithic_sequence | 0.685 | 0.243 | 0.594 | 0.634 |

Interpretation:
- Removing the fast state hurt by about **0.013** log-loss points.
- Removing the slow state hurt by about **0.008** log-loss points.
- The full latent model beat the static baseline by about **0.030** log-loss points.
- The full latent model beat the monolithic sequence baseline by about **0.055** log-loss points.

That is enough to say the slow/fast state framing is doing real work in this synthetic regime.

### Probe target

On the auxiliary probe target, `latent_full` also beat the simple current-touch model by about **0.037** log-loss points.

That matters because the paper does not want the latent state to be useful for one binary target only. The probe head is the sanity check that the state carries reusable structure.

### One honest ablation lesson

In this toy regime, collapsed slow pooling was nearly tied with the fuller slow representation.

That is not an embarrassment. It is the correct benchmark attitude.

A design claim should not survive because it sounds deep. It should survive because it improves performance on held-out data. In some real domains, regime-aware separation may pay rent; in others, it may not. The framework is set up to make that answer empirical.

## 5. IPS / policy evaluation

The IPS toy demo landed at:
- IPS estimate: **0.612**
- true target-policy value: **0.609**
- naive matched mean: **0.577**

That is close enough to make the operational point visible:
- logged propensities let you do offline policy evaluation
- absent propensities, you are ranking and simulating, not making clean causal claims

## 6. What this means for real-world use

### GTM / sales systems
This is the most obvious fit because the paper is already written in that register.
- person-side object: buyer, founder, operator, champion
- proposition: message, offer, timing, framing, sequence step
- outputs: reply, meeting, objection class, delay, stage advance

The toy results suggest the framework is worthwhile when both durable account/person structure and recent interaction state matter.

### Support and escalation
- person-side object: user, admin, account health
- proposition: response template, escalation, compensation, channel switch
- outputs: resolution, re-open risk, escalation risk, CSAT

The categorical pooling logic is especially natural here because the raw traces often are repeated discrete tags and themes.

### Retention / lifecycle
- person-side object: user or account with durable preferences and current activation state
- proposition: onboarding nudge, pricing experiment, education message
- outputs: activation, usage depth, renewal, churn risk

The slow/fast split is a good fit when “what this account is generally like” and “what is live this week” are both predictive.

### Recruiting
- person-side object: candidate under a role and process stage
- proposition: outreach framing, comp framing, timing, interviewer sequencing
- outputs: reply, process advance, acceptance, delay

### Care management or outreach
The algebra still fits, but the bar is much higher.
- more regulation
- more ethics
- more intervention risk
- stronger need for calibration and causal discipline

That is exactly where the paper’s warning matters: ranking is not causality.

## 7. What not to overclaim

This project does **not** prove:
- that the full phenomenal state has been recovered,
- that the toy latent state is the unique minimal state for real humans,
- or that proposition search is automatically causal.

What it **does** prove is smaller and useful:
- the operational equations can be implemented cleanly,
- the patched objective and slow-update rules are the right ones to keep,
- and the framework is packaged cleanly enough that another team can swap in real encoders, real history, and real policy logging.
