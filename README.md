# ReikaSystem v11.0

External Control Framework for Large Language Models

**Author**: Takayuki Ishii (Independent Researcher)

## What is ReikaSystem?

ReikaSystem is an external control protocol for LLMs designed to prevent **Halation**: outputs that are fluent and internally coherent, yet structurally wrong.

ReikaSystem treats the LLM as an untrusted candidate generator and relocates decision authority to an external, deterministic control layer.

## Core Principles

- **Empty Set Principle**: LLM ∩ GroundTruth = ∅
- **RF/GM Risk Model**: Detects imbalance between rhetorical fluency and grounding mass
- **HOLD-as-Success**: Abstention is a positive safety outcome

## Files

- `reika_system_v11_4.py` — Reference implementation (prototype)
- `reika_v11_updated.tex` — Paper source (LaTeX)

## License

MIT
