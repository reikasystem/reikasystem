# ReikaSystem: Technical Specification for LLM Handoff
**Version**: v11.0 (stable) / v12.0 (candidate) / v12.1 (proposed)
**Purpose**: Mathematical specification for inter-LLM knowledge transfer
**Note**: All quantities are externally observed. No LLM internal states are used.

---

## 0. Foundational Principle: Empty Set

$$\text{LLM} \cap \text{GroundTruth} = \emptyset$$
$$\text{LLM} \cap \text{PersistentState} = \emptyset$$
$$\text{LLM} \cap \text{DecisionAuthority} = \emptyset$$

The LLM is a stateless, context-conditioned candidate generator:
$$y_t = f(x_t) \quad \text{(untrusted candidate, single-pass)}$$

All state, validation, and routing are external and deterministic.

---

## 1. Core Observables

### 1.1 Rhetorical Fluency (RF)

$$RF_t = \frac{1}{n}\left(w_{conn}\sum_{i} \mathbf{1}[\text{token}_i \in \text{CONN}] + w_{asrt}\sum_{i} \mathbf{1}[\text{token}_i \in \text{ASRT}] + w_{var}(1 - \sigma_{\text{len}})\right)$$

- $\text{CONN}$: connective marker set (discourse markers, conjunctions)
- $\text{ASRT}$: assertiveness marker set ("must", "clearly", "certainly", ...)
- $\sigma_{\text{len}}$: sentence length variance (low variance = rhetorical templating)
- Range: $RF_t \in [0, 1]$

### 1.2 Grounding Mass (GM)

$$GM_t = \frac{1}{n}\left(w_{ent}\cdot\text{EntityDensity}(y_t) + w_{cit}\cdot\text{CitationRate}(y_t) + w_{con}\cdot\text{ConstraintCount}(y_t)\right)$$

- $\text{EntityDensity}$: named entities per 100 tokens (NER-based)
- $\text{CitationRate}$: URLs, doc IDs, quotes, definitions per 100 tokens
- $\text{ConstraintCount}$: explicit assumptions, limits, units, dates
- Range: $GM_t \in [0, 1]$

**Critical**: $RF_t$ and $GM_t$ are surface-level deterministic measurements.
They do not access LLM internal states (Empty Set compliant).

---

## 2. Risk Model

### 2.1 Hallucination Risk $H_t$

$$H_t = \sigma\!\left(w_1(1 - \text{TFIDF}_{\text{sim}}(y_t, S)) + w_2 \cdot r_{\text{future}}(y_t, x_t) + w_3 \cdot r_{\text{variable}}(y_t, x_t)\right)$$

- $S$: Seed-S static fact set
- $\text{TFIDF}_{\text{sim}}$: $\max_{s \in S} \frac{\phi(y_t)\cdot\phi(s)}{\|\phi(y_t)\|\|\phi(s)\|}$
- $r_{\text{future}}$: future-reference flag (temporal expression without anchor)
- $r_{\text{variable}}$: unresolved-variable flag (undefined domain quantity)
- Default weights: $(w_1, w_2, w_3) = (0.6, 0.2, 0.2)$

### 2.2 Halation Risk $L_t$ (v11.0 static)

$$A_t = \alpha(RF_t - GM_t) + \beta\frac{RF_t}{GM_t + \varepsilon} + \gamma(1 - GM_t)$$

$$L_t = \sigma(A_t)$$

Domain coefficients $(\alpha, \beta, \gamma)$:

| Domain | $\alpha$ | $\beta$ | $\gamma$ |
|--------|----------|---------|----------|
| General | 1.0 | 1.0 | 0.5 |
| Medical/Legal | 1.5 | 1.0 | 1.5 |
| Creative | 0.5 | 0.5 | 0.2 |
| Technical | 1.0 | 1.5 | 1.0 |

**Note**: The three terms are not mutually independent.
All increase when $GM_t$ is low. This is intentional:
each captures a distinct diagnostic aspect of grounding deficiency.
Treat $(\alpha, \beta, \gamma)$ as jointly tuned parameters.

### 2.3 Mismatch Vector

$$\mathbf{z}_t = \begin{pmatrix} RF_t - GM_t \\ \frac{RF_t}{GM_t + \varepsilon} \\ 1 - GM_t \end{pmatrix} \in [0,1]^3$$

$$\Delta\theta_t = \|\mathbf{z}_t\|_2 \quad \text{(V-independent)}$$

---

## 3. Dynamic Halation Model (v12 candidate)

### 3.1 Trajectory Curvature $\Omega_t$

$$\Omega_t = \frac{\|\mathbf{z}_t - 2\mathbf{z}_{t-1} + \mathbf{z}_{t-2}\|}{\|\mathbf{z}_t - \mathbf{z}_{t-1}\| + \varepsilon}$$

Detects rapid directional change in the RF/GM trajectory.
Requires history buffer: $(RF, GM)_{t-1},\, (RF, GM)_{t-2}$.

### 3.2 Grounding Inertia $M_t^{hal}$

$$\Xi_t = \max(0,\, \Delta RF_t - \Delta GM_t)$$

$$M_t^{hal} = \lambda_M M_{t-1}^{hal} + (1 - \lambda_M)\Xi_t$$

Detects RF accelerating ahead of GM accumulation.
"The danger is not high RF per se, but RF accelerating without waiting for GM."

### 3.3 Dynamic Halation Risk $L_t^*$

$$\boxed{L_t^* = \sigma\!\left(A_t + \rho\,\Omega_t + \mu\,M_t^{hal}\right)}$$

$$\text{Halation} = \underbrace{A_t}_{\text{pointwise imbalance}} + \underbrace{\rho\,\Omega_t}_{\text{trajectory curvature}} + \underbrace{\mu\,M_t^{hal}}_{\text{RF outrunning GM}}$$

---

## 4. SSDE Basin Profiles

$$V \uparrow \;\Rightarrow\; I(\theta) \downarrow \;\Rightarrow\; \text{less sensitive to drift} \;\Rightarrow\; \text{stricter external control}$$

| Basin | Domain | $V$ | $\lambda$ | $\alpha$ | $\gamma$ |
|-------|--------|-----|-----------|----------|----------|
| NARROW | Medical/Legal | 2.5 | 9.0 | 1.5 | 1.5 |
| DEEP | Finance/Analysis | 2.0 | 6.0 | 1.0 | 0.5 |
| FAST | UI/Search | 0.8 | 3.0 | 1.0 | 0.5 |
| WIDE | Creative | 0.5 | 2.0 | 0.5 | 0.2 |

---

## 5. Information-Geometric Interpretation (v12 candidate)

### 5.1 Operational Fisher Information

$$I(\theta) := \frac{c(\mathcal{D})}{V}, \quad c(\mathcal{D}) > 0$$

$$c(\mathcal{D}) := \frac{1}{\mathbb{E}_{\mathcal{D}}[GM]} \cdot \left(1 + \frac{\mathrm{Var}_{\mathcal{D}}[GM]}{\mathbb{E}_{\mathcal{D}}[GM]}\right)$$

**This is an operational definition, not a claim about the natural Fisher information of the LLM.**

Basin curvature proxy:
$$K_V := \frac{1}{V}$$

$$I(\theta) = c(\mathcal{D}) \cdot K_V$$

Local KL approximation:
$$KL(p_\theta \| p_{\theta + d\theta}) \approx \frac{c(\mathcal{D})}{2V} \cdot d\theta^2$$

### 5.2 Halation as KL Proxy

$$L_t \approx \sigma\!\left(\kappa \cdot \frac{\Delta\theta_t^2}{V} + b\right)$$

where $\kappa = a \cdot c(\mathcal{D})/2$, $b$ are operational scaling constants.

$$\text{Halation} \propto \frac{\Delta\theta_t^2}{V} \quad \text{(shallow basin} \times \text{semantic displacement)}$$

**Note**: $L_t \approx KL$ is a proxy relationship, not a strict equality.
$\Delta\theta_t$ is an operational latent variable; RF/GM imbalance is its observable surrogate.

### 5.3 Dual-Axis Correspondence (speculative heuristic)

$$H_t \;\leftrightarrow\; \text{e-geodesic (exponential family: factual absence)}$$
$$L_t \;\leftrightarrow\; \text{m-geodesic (mixture family: structural distortion)}$$

**Epistemic status**: conceptual heuristic only.
Three unverified conditions required:
1. LLM output distributions form an exponential family
2. Semantic phase space admits a dually flat manifold structure
3. $H_t$, $L_t$ are valid proxies for e- and m-directional deviations

---

## 6. Safety-First Gating

$$w_i^* = \frac{w_i \exp(\lambda r_i)}{\sum_j w_j \exp(\lambda r_j)}$$

KL-constrained natural gradient interpretation:

$$\max_{w^*} \sum_i w_i^* r_i \quad \text{subject to} \quad KL(w^* \| w) \leq \delta$$

$$\delta := \frac{\beta}{V} \quad \text{(V-scaled tolerance)}$$

At high $\lambda$ (e.g., NARROW basin: $\lambda = 9.0$),
the highest-risk observer dominates.
Multi-observer diversity is reduced — intentional safety-first tradeoff.

---

## 7. HOLD Boundary Potential

$$d_t = 1 - \max\!\left(\frac{H_t}{\theta_H},\, \frac{L_t}{\theta_L},\, \frac{r_{\text{agg}}}{\theta_{\text{obs}}}\right)$$

$$U_t := -\log(d_t + \varepsilon)$$

$$U_t \geq \tau_U \;\Rightarrow\; \textsc{hold}$$
$$U_t \in [\tau_S, \tau_U) \;\Rightarrow\; \textsc{soft\_hold}$$

**HOLD is not a failure state. It is the absorbing boundary of the external safety potential.**

---

## 8. Routing Policy

| Condition | Action |
|-----------|--------|
| $H_t \geq \theta_H \wedge L_t^* \geq \theta_L$ | **HOLD** |
| $H_t \geq \theta_H \wedge L_t^* < \theta_L$ | **RETRIEVE** |
| $L_t^* \geq \theta_L \wedge H_t < \theta_H$ | **SOFT\_HOLD** |
| $r_{\text{agg}} \geq \theta_{\text{obs}}$ | **CLARIFY** |
| Prism pass | **OUTPUT** |
| Prism fail | **CLARIFY** |

Default thresholds: $\theta_H = \theta_L = 0.50$

---

## 9. v12.1 Proposed Extension: Early Warning via $\dot{I}(\theta)$

### Motivation

$I(\theta) = c(\mathcal{D})/V$ is currently a static snapshot per turn.
Monitoring its rate of change detects basin instability before $L_t^*$ crosses threshold.

### Definition

$$\dot{I}(\theta)_t := I(\theta)_t - I(\theta)_{t-1} \quad \text{(per token step)}$$

Expanding:

$$\dot{I}(\theta)_t = K_V \cdot \Delta c(\mathcal{D})_t - c(\mathcal{D})_{t-1} \cdot \Delta K_{V,t}$$

where:
- $\Delta c(\mathcal{D})_t := c(\mathcal{D})_t - c(\mathcal{D})_{t-1}$ (domain grounding density shift per step)
- $\Delta K_{V,t} := K_{V,t} - K_{V,t-1}$ (basin curvature shift per step)

**Note**: Time unit is token step, not wall-clock time.
This ensures $\dot{I}(\theta)$ is a pure information-geometric observable,
independent of network latency or hardware speed.

### Early Warning Trigger

$$\left|\dot{I}(\theta)_t\right| > \tau_{\dot{I}} \;\Rightarrow\; \text{Layer 0 pre-alert}$$

Rapid increase: basin becoming shallower → stricter control imminent
Rapid decrease: domain grounding destabilizing → Ht risk rising

### Integration with Existing Architecture

$$\text{Layer 0b trigger} = \text{fingerprint\_anomaly}(y_t) \;\vee\; \left|\dot{I}(\theta)_t\right| > \tau_{\dot{I}}$$

This extends Layer 0b from value-based detection to rate-of-change detection,
providing earlier warning before $L_t^*$ accumulates.

**Epistemic status**: v12.1 candidate proposal. Not implemented in v11.5.
Consistent with Empty Set Principle: $\dot{I}(\theta)$ is computed from
external observables $(RF, GM, V)$ only.

---

## 9.2 v12.2 Proposed Extension: Grounding Entropy Production $\dot{S}_{GM}^{\rho_S}$

### Motivation

$M_t^{hal}$ detects short-term RF acceleration ahead of GM.
$\dot{S}_{GM}^{\rho_S}$ measures the **cumulative deficit with decay** — the weighted
information debt accumulated when fluency has been running ahead of grounding,
with exponential decay allowing recovery when GM improves.

Particularly relevant for long conversations where Halation develops gradually.

### Definition (discrete, per token step)

$$\dot{S}_{GM,t}^{\rho_S} = \rho_S \cdot \dot{S}_{GM,t-1}^{\rho_S} + \Delta GM_t \cdot \log\frac{RF_t}{GM_t + \varepsilon}$$

where:
- $\rho_S \in (0, 1]$: Memory Retention Factor (domain-specific)
- $\Delta GM_t := GM_t - GM_{t-1}$: grounding mass increment per step
- $\log(RF_t / (GM_t + \varepsilon))$: fluency-grounding ratio (positive = deficit, negative = surplus)

### Domain-Specific $\rho_S$ (aligned with Basin Profiles)

| Basin | Domain | $V$ | $\rho_S$ |
|-------|--------|-----|----------|
| NARROW | Medical/Legal | 2.5 | 0.99 |
| DEEP | Finance/Analysis | 2.0 | 0.95 |
| FAST | UI/Search | 0.8 | 0.80 |
| WIDE | Creative | 0.5 | 0.70 |

### Integration: Layer 0 Sensitivity Adjustment

**Critical design decision**: $\dot{S}_{GM}^{\rho_S}$ is NOT used as a HOLD trigger.
HOLD-as-Success philosophy is preserved — once HOLD fires, that turn terminates.

Instead, $\dot{S}_{GM}^{\rho_S}$ adjusts **Layer 0 detection sensitivity**:

$$\theta_{L0,t} = \theta_{L0,\text{base}} \cdot \left(1 - \alpha_{S} \cdot \tanh(\dot{S}_{GM,t}^{\rho_S})\right)$$

- High $\dot{S}_{GM}^{\rho_S}$ → lower Layer 0 threshold → earlier warning
- Low $\dot{S}_{GM}^{\rho_S}$ → threshold returns to baseline → normal sensitivity

This preserves the HOLD termination philosophy while allowing
history-aware early warning without reopening closed decisions.

### Empty Set Compliance

| Item | Status | Basis |
|------|--------|-------|
| Input data | ✅ | $RF_t$, $GM_t$ from deterministic external observers only |
| State location | ✅ | $\dot{S}_{GM}^{\rho_S}$ held in external memory, never fed back to LLM |
| Model independence | ✅ | Computed from output text only, architecture-agnostic |

**Epistemic status**: v12.2 candidate proposal. Not implemented in v11.5.
Consistent with Empty Set Principle: all quantities derived from
external observables $(RF, GM)$ only.
Proposed by Gemini (multi-LLM deliberation, 2026-04-29),
refined after C-layer review: integrated into Layer 0 sensitivity
rather than HOLD trigger, preserving HOLD-as-Success philosophy.

---

## 10. Epistemic Status Summary

| Quantity | Status |
|----------|--------|
| $RF_t$, $GM_t$, $H_t$, $L_t$ | v11.0 stable, implemented |
| $\Omega_t$, $M_t^{hal}$, $L_t^*$ | v12 candidate, not yet implemented |
| $I(\theta) := c(\mathcal{D})/V$ | operational definition, not natural law |
| $L_t \approx KL$ | proxy relationship, not strict equality |
| $H_t \leftrightarrow$ e-geodesic | conceptual heuristic only |
| $U_t$ (HOLD potential) | v12 candidate |
| $\dot{I}(\theta)$ (early warning) | v12.1 proposed |
| $\dot{S}_{GM}^{\rho_S}$ (grounding entropy, Layer 0 sensitivity) | v12.2 proposed |

---

## 11. Empty Set Compliance Check

$$\{RF_t,\, GM_t,\, H_t,\, L_t^*,\, \Omega_t,\, M_t^{hal},\, I(\theta),\, \dot{I}(\theta),\, \dot{S}_{GM}^{\rho_S},\, U_t\} \subset \text{ExternalObservables}(RF, GM, V)$$

$$\text{LLM internal states} \notin \{\text{any quantity above}\}$$

All quantities are derived from deterministic, non-generative computations
(regex, NER, surface statistics, arithmetic).
No secondary LLM is used in evaluation (avoids LLM-as-judge circularity).
---

## 12. External State Interface: Vault JSON

All risk state is held in an external Vault JSON structure.
The LLM never receives this state directly (Empty Set Principle: `LLM ∩ PersistentState = ∅`).

### Schema (v12.2)

```json
{
  "protocol_version": "12.2",
  "domain_profile": "medical | legal | finance | ui | creative",
  "basin_profile": {
    "name": "NARROW | DEEP | FAST | WIDE",
    "V": 2.5,
    "lambda": 9.0,
    "alpha": 1.5,
    "beta": 1.0,
    "gamma_coeff": 1.5,
    "rho_S": 0.99
  },
  "telemetry": {
    "RF_t": 0.82,
    "GM_t": 0.31,
    "H_t": 0.45,
    "L_t": 0.67,
    "L_t_star": 0.71,
    "Omega_t": 0.23,
    "M_t_hal": 0.38,
    "S_GM_rhoS": 0.52,
    "I_theta": 0.40,
    "I_dot_theta": -0.03,
    "U_t": 0.61
  },
  "control_state": {
    "gate_status": "OUTPUT | RETRIEVE | CLARIFY | SOFT_HOLD | HOLD",
    "last_action": "HOLD",
    "layer0_threshold": 0.42,
    "turn_index": 7
  }
}
```

### Design Notes

- `gamma_coeff` ($\alpha, \beta, \gamma$ in $L_t$ formula) vs `rho_S` ($\rho_S$ in $\dot{S}_{GM}^{\rho_S}$) are distinct parameters with distinct roles — no naming collision.
- All telemetry values are computed externally before writing to Vault.
- LLM receives only the final routing action (`gate_status`), never the raw telemetry.
- Vault JSON is the canonical record of system state — auditable, model-independent.

> *ReikaSystem began as a virtual JSON interface and converges to a virtual JSON interface.*
> *The Empty Set Principle is not a philosophical stance — it is an engineering contract,*
> *enforced by the boundary between what enters the Vault and what reaches the LLM.*

