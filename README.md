[README.md](https://github.com/user-attachments/files/26870236/README.md)
# ReikaSystem v11.0

**External Control Framework for Large Language Models**

**Author**: Takayuki Ishii (Independent Researcher)  
**arXiv**: (submission pending)  
**License**: MIT

---

## What is ReikaSystem?

ReikaSystem is an external control protocol for LLMs designed to prevent **Halation**: outputs that are fluent and internally coherent, yet structurally wrong due to missing dependencies, drifting definitions, or insufficient grounding.

ReikaSystem treats the LLM as an **untrusted candidate generator** and relocates decision authority to an external, deterministic, auditable control layer.

---

## Core Principles

- **Empty Set Principle**: LLM ∩ GroundTruth = ∅ / LLM ∩ PersistentState = ∅ / LLM ∩ DecisionAuthority = ∅
- **RF/GM Risk Model**: Detects imbalance between rhetorical fluency (RF) and grounding mass (GM)
- **HOLD-as-Success**: Abstention is a positive safety outcome, not a failure
- **Layer 0**: Deterministic pre- and mid-generation fingerprint detection for early exit
- **Safety-First Gating**: Exponential reweighting of observer scores toward safety

---

## Halation vs. Hallucination

| | Hallucination | Halation |
|---|---|---|
| Cause | Knowledge gap / fabrication | RF/GM structural imbalance |
| Surface | Factually wrong | Fluent, plausible, locally coherent |
| Fix | Retrieval (RAG) | External control + routing |

---

## Routing Actions

| Action | Condition |
|---|---|
| OUTPUT | Low Ht, low Lt, Prism pass |
| RETRIEVE | High hallucination risk (Ht) |
| CLARIFY | Structural ambiguity or observer concern |
| HOLD | Dual high-risk — safe abstention |
| SOFT_HOLD | High Lt only — annotated output with [requires-verification] |

---

## Files

- `reika_system_v11_4.py` — Reference implementation (prototype, not production)
- `reika_v11_tikz_v9.tex` — Paper source (LaTeX, v11.0 + v11.5 additions)

---

## Position

This is a **reference implementation / prototype** intended to illustrate the external control flow described in the ReikaSystem v11.0 paper.

- NOT a production deployment package
- NOT a complete safety solution
- Feature proxies (RF/GM, observers) are deliberately lightweight and require domain-specific calibration

---

## Citation

```
@misc{ishii2026reikasystem,
  title={ReikaSystem v11.0: A Unified External Control Framework for Large Language Models},
  author={Takayuki Ishii},
  year={2026},
  note={arXiv preprint (submission pending)}
}
```

---

## License

MIT
