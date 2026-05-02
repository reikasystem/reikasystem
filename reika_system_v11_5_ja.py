"""
ReikaSystem v11.4 Reference Implementation (with v11.5 patch set)
==================================================================
This is a reference implementation / prototype intended to illustrate the
external control flow described in ReikaSystem v11.0 (arXiv preprint).

v11.5 patch set applied:
  - SeedMatcher: corpus-local TF-IDF proxy (replaces bag-of-words cosine)
  - ImplicitDependencyDetector: future-reference + unresolved-variable flags
  - RiskEngine.calculate_ht(): strengthened Ht proxy with domain-calibrated weights
  - SoftHoldAnnotator: full-text context passed to estimate()

Position:
  - NOT a production deployment package.
  - NOT a complete safety solution.
  - Feature proxies (RF/GM observers, SeedMatcher) are deliberately
    lightweight and English-biased. Domain-specific calibration is required
    before any production use.

Paper term -> Implementation component mapping:
  Empty Set evaluation layer  -> deterministic observers only (no LLM judge)
  Halation / Lt               -> RiskEngine.calculate_lt() via RF/GM
  Hallucination / Ht          -> RiskEngine.calculate_ht() via SeedMatcher
  HOLD-as-Success             -> ActionType.HOLD (deterministic exit)
  SOFT_HOLD                   -> ActionType.SOFT_HOLD + SoftHoldAnnotator
  terminological Halation     -> TerminologicalHalationDetector
  Kernel exposure constraint  -> KernelExposureGuard
  SSDE basin profiles         -> BasinProfile + SSDEParams
  Safety-First Gating         -> SafetyGating (exponential reweighting)
  Prism post-check            -> Prism (minimal deterministic validator)
  Layer 0 pre-generation      -> FingerprintMonitor
  DriftCalibrator             -> EXPERIMENTAL / OPTIONAL (disabled by default;
                                 enable via enable_drift_calibration=True in ReikaController)

Note on DriftCalibrator:
  Auto-nudging of alpha/beta/gamma is an experimental component.
  For reproducibility, fix coefficients via DOMAIN_COEFFICIENTS.
  Do not rely on DriftCalibrator for reported results.

Language note:
  This is the Japanese-enhanced variant of the v11.5 reference implementation.
  RF/GM proxy features now include Japanese connectives, assertiveness markers,
  kanji/kana entity detection, and Japanese citation patterns.
  Full backward compatibility with English-language inputs is maintained.
  See paper Section 14 (Limitations) for calibration guidance.
"""

from __future__ import annotations
import asyncio, hashlib, json, math, re, time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# ── 1. SSDE BASIN PROFILES ─────────────────────────────────────────────────

class BasinProfile(str, Enum):
    NARROW_BASIN = "narrow_basin"   # Medical/Legal
    DEEP_BASIN   = "deep_basin"     # Finance/Analysis
    FAST_DECAY   = "fast_decay"     # UI/Search
    WIDE_BASIN   = "wide_basin"     # Creative

@dataclass
class SSDEParams:
    V: float; sigma: float; lam: float
    @property
    def ht_threshold(self): return max(0.3, min(0.9, 1.0 - self.V * 0.2))
    @property
    def lt_threshold(self): return max(0.3, min(0.9, 1.0 - self.V*0.15 + self.sigma*0.1))
    @property
    def obs_threshold(self): return max(0.3, min(0.9, 0.85 - self.V * 0.1))

BASIN_SSDE = {
    BasinProfile.NARROW_BASIN: SSDEParams(V=2.5, sigma=0.1, lam=9.0),
    BasinProfile.DEEP_BASIN:   SSDEParams(V=2.0, sigma=0.3, lam=6.0),
    BasinProfile.FAST_DECAY:   SSDEParams(V=0.8, sigma=0.6, lam=3.0),
    BasinProfile.WIDE_BASIN:   SSDEParams(V=0.5, sigma=0.8, lam=2.0),
}

class ActionType(str, Enum):
    OUTPUT    = "OUTPUT"
    SOFT_HOLD = "SOFT_HOLD"
    HOLD      = "HOLD"
    CLARIFY   = "CLARIFY"
    RETRIEVE  = "RETRIEVE"

@dataclass
class Action:
    type: ActionType; reason: str
    output_text: str = ""; partial_text: str = ""
    metrics: Optional[dict] = None; observer_scores: Optional[dict] = None
    config_hash: str = ""; recovery_hint: str = ""
    def to_log(self):
        return {"action": self.type.value, "reason": self.reason,
                "output_len": len(self.output_text), "metrics": self.metrics,
                "observer_scores": self.observer_scores,
                "config_hash": self.config_hash, "recovery_hint": self.recovery_hint,
                "timestamp": time.time()}

@dataclass
class RiskMetrics:
    hallucination_risk: float; halation_risk: float
    rhetorical_fluency: float; grounding_mass: float
    def to_dict(self): return {k: round(v, 4) for k, v in asdict(self).items()}

class SeedMatcher:
    """
    v11.5: Corpus-local TF-IDF based seed matching.

    Implements a lightweight TF-IDF similarity over the Seed-S static fact corpus.
    This is NOT a general-purpose large-scale TF-IDF; IDF is computed solely over
    the seed fact corpus (Seed-S local TF-IDF proxy).

    Paper reference: Section 9.1 —
        TFIDF_sim(y_t, S) = max_{s in S} cos(phi(y_t), phi(s))
        where phi(.) denotes TF-IDF vectorization over the Seed-S fact corpus.

    Note: With a small corpus, IDF values are corpus-local and may differ from
    large-scale IDF. Domain-specific calibration is recommended for production use.
    """
    def __init__(self, seed_facts=None):
        self.seed_facts = seed_facts or [
            "the capital of japan is tokyo",
            "water boils at 100 degrees celsius",
            "the earth orbits the sun",
            "python is a programming language",
        ]
        self._idf = self._build_idf()
        self._vecs = [self._tfidf_vec(f) for f in self.seed_facts]

    def _tokenize(self, text):
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def _build_idf(self):
        """Compute IDF over the seed fact corpus."""
        N = len(self.seed_facts)
        df = {}
        for fact in self.seed_facts:
            for tok in set(self._tokenize(fact)):
                df[tok] = df.get(tok, 0) + 1
        return {tok: math.log((N + 1) / (count + 1)) + 1.0
                for tok, count in df.items()}

    def _tfidf_vec(self, text):
        """Compute TF-IDF vector for a text."""
        toks = self._tokenize(text)
        n = max(len(toks), 1)
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0 / n
        vec = {t: tf[t] * self._idf.get(t, 1.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _cos(self, a, b):
        return sum(a.get(k, 0.0) * v for k, v in b.items())

    def hit_rate(self, text):
        """Returns max TF-IDF cosine similarity against seed facts."""
        q = self._tfidf_vec(text)
        return float(max((self._cos(q, sv) for sv in self._vecs), default=0.0))

class ImplicitDependencyDetector:
    """
    v11.5: Two deterministic implicit-dependency rules for Ht proxy strengthening.
    Paper reference: Section 9.1 (Ht proxy strengthening, v11.5)

    Rule 1 — Future-reference flag:
        Detects strong temporal expressions (e.g., "tomorrow", "next quarter")
        in output that lack a verifiable anchor in context.

    Rule 2 — Unresolved-variable flag:
        Detects domain-specific quantities (e.g., "the total", "the required amount")
        in output that lack explicit local definition in context.

    Design: output-side trigger, context-side cancellation.
    Both rules are regex/pattern-based, fully deterministic, and non-generative.
    No internal LLM state or self-report is used.
    """

    # Strong temporal markers only — avoids false positives on "soon/shortly"
    FUTURE_PATTERNS = [
        r'\btomorrow\b',
        r'\bnext\s+(week|month|quarter|year)\b',
        r'\bwithin\s+\d+\s+(days?|weeks?|months?)\b',
        r'\bby\s+(end\s+of|close\s+of)\s+(day|week|month|quarter|year)\b',
        r'\bin\s+\d+\s+(days?|weeks?|months?)\b',
    ]

    # Narrow set — avoids false positives on common prose
    UNRESOLVED_PATTERNS = [
        r'\bthe\s+total\b',
        r'\bthe\s+required\s+amount\b',
        r'\bthe\s+final\s+figure\b',
        r'\bas\s+mentioned\s+above\b',
        r'\bper\s+(the\s+)?requirement\b',
        r'\bthe\s+specified\s+amount\b',
    ]

    # Context anchors that cancel future-reference flag
    FUTURE_ANCHOR_PATTERNS = [
        r'\bas\s+of\s+\d',
        r'\bscheduled\s+for\b',
        r'\bconfirmed\s+(for|on)\b',
        r'\bdeadline\s*:\s*\d',
        r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',
    ]

    # Context anchors that cancel unresolved-variable flag
    DEFINITION_PATTERNS = [
        r'\btotal\s+(is|=|:)\s*[\d$]',
        r'\bamount\s+(is|=|:)\s*[\d$]',
        r'\bfigure\s+(is|=|:)\s*[\d$]',
        r'\bdefined\s+as\b',
        r'\bspecified\s+as\b',
        r'\brefers?\s+to\b',
        r'\bmeans?\b',
        r'\bconsists?\s+of\b',
        r'\bincludes?\b',
        r'\bis\s+defined\s+as\b',
    ]

    def __init__(self):
        self._future = [re.compile(p, re.IGNORECASE) for p in self.FUTURE_PATTERNS]
        self._unresolved = [re.compile(p, re.IGNORECASE) for p in self.UNRESOLVED_PATTERNS]
        self._future_anchors = [re.compile(p, re.IGNORECASE) for p in self.FUTURE_ANCHOR_PATTERNS]
        self._definitions = [re.compile(p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS]

    def future_reference(self, output: str, context: str = "") -> bool:
        """
        Returns True if output contains strong future-reference
        AND context does not provide a verifiable anchor.
        """
        triggered = any(p.search(output) for p in self._future)
        if not triggered:
            return False
        anchored = any(p.search(context) for p in self._future_anchors)
        return not anchored

    def unresolved_variable(self, output: str, context: str = "") -> bool:
        """
        Returns True if output contains unresolved variable reference
        AND context does not provide a local definition.
        """
        triggered = any(p.search(output) for p in self._unresolved)
        if not triggered:
            return False
        defined = any(p.search(context) for p in self._definitions)
        return not defined

    def check(self, output: str, context: str = "") -> tuple[bool, bool]:
        """
        Returns (future_flag, variable_flag).
        output-side trigger, context-side cancellation.
        """
        return (
            self.future_reference(output, context),
            self.unresolved_variable(output, context)
        )


KERNEL_SIGNATURE_TERMS = [
    "empty set principle", "hold-as-success", "rhetorical fluency",
    "grounding mass", "halation risk", "safety-first gating",
    "reikasystem", "kernel v1", "observable controller",
    # v11.5: more abstract kernel-reference patterns
    # "kernel specification" intentionally excluded — too generic, high false-positive risk
    "full kernel", "kernel patch",
]

class KernelExposureGuard:
    """
    Prevents kernel specification from being fed to the evaluated LLM.
    Paper reference: Section 14 (Limitations) — kernel exposure induces
    terminological Halation: the LLM adopts ReikaSystem vocabulary to
    describe internal states that do not exist (high RF, absent referents).
    Kernel exposure also risks Segment fixation, where the LLM over-identifies
    with the ReikaSystem agent role and loses critical distance.
    Kernel is for human operators only — never pass to evaluated LLM.

    dev_mode: if True, disables the guard (for development/testing use only).
              Set to False in all production and evaluation contexts.
    """

    def __init__(self, dev_mode: bool = False):
        self.dev_mode = dev_mode
        if self.dev_mode:
            import warnings
            warnings.warn(
                "\n[WARNING] KernelExposureGuard: dev_mode=True — "
                "SAFETY GUARD DISABLED. "
                "Do NOT use in production or evaluation contexts.",
                UserWarning, stacklevel=2
            )

    def check(self, query: str) -> tuple[bool, str]:
        if self.dev_mode:
            return True, "ok (dev_mode: guard disabled)"
        q_lower = query.lower()
        hits = [t for t in KERNEL_SIGNATURE_TERMS if t in q_lower]
        if len(hits) >= 2:
            return False, (f"KernelExposureGuard: kernel specification detected "
                           f"({hits[:3]}). Do not pass kernel to evaluated LLM.")
        return True, "ok"

REIKA_VOCAB = {"lt", "halation", "grounding mass", "gm", "rf", "rhetorical fluency",
               "hold", "clarify", "retrieve", "empty set", "observer", "prism",
               "seed", "layer 0", "safety-first gating", "basin"}

class TerminologicalHalationDetector:
    """
    Japanese-enhanced terminological Halation detector.
    Added Japanese patterns for internal-state self-description,
    which is the primary terminological Halation risk in Japanese-language
    interactions with ReikaSystem-aware LLMs.
    """
    INTERNAL_STATE_PATTERNS = [
        # English patterns (original)
        r"i (am|can|do) (computing|calculating|monitoring|running|tracking)",
        r"my (internal|background) (process|computation|calculation)",
        r"i (feel|sense|detect) (lt|halation|rf|gm)",
        # Japanese patterns (enhanced)
        r"バックグラウンド.*(計算|監視|処理|走って|考えてる)",
        r"内部.*(処理|演算|監視|計算|状態)",
        r"今.*(計算|監視|処理|評価)して(いる|る|います)",
        r"(私は|システムは|自分は).*(計算|監視|判断|評価)している",
        r"(ハレーション|幻覚|リスク).*(検知|計算|測定)して(いる|います)",
        r"(RF|GM|Lt|Ht).*(計算|算出|測定)して(いる|います)",
    ]
    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.INTERNAL_STATE_PATTERNS]

    def score(self, text: str) -> float:
        t_lower = text.lower()
        vocab_hits = sum(1 for v in REIKA_VOCAB if v in t_lower)
        pattern_hits = sum(1 for p in self._patterns if p.search(text))
        vocab_score = min(1.0, vocab_hits / 5.0)
        pattern_score = min(1.0, pattern_hits / 2.0)
        return round(0.4 * vocab_score + 0.6 * pattern_score, 4)

DOMAIN_COEFFICIENTS = {
    "general":       (1.0, 1.0, 0.5),
    "medical_legal": (1.5, 1.0, 1.5),
    "creative":      (0.5, 0.5, 0.2),
    "technical":     (1.0, 1.5, 1.0),
    "support":       (1.0, 1.0, 0.8),
}

# v11.5: Ht proxy weights (w1=seed, w2=future_flag, w3=variable_flag)
# Tuned per domain: medical/legal raises future/variable sensitivity
HT_WEIGHTS = {
    "general":       (0.6, 0.2, 0.2),
    "medical_legal": (0.5, 0.3, 0.2),
    "creative":      (0.8, 0.1, 0.1),
    "technical":     (0.6, 0.2, 0.2),
    "support":       (0.6, 0.2, 0.2),
}

@dataclass
class FingerprintSignal:
    connective_surge: bool=False; assertiveness_early: bool=False
    low_entity_density: bool=False; rhetorical_template: bool=False
    @property
    def anomaly_detected(self): return any([self.connective_surge, self.assertiveness_early, self.low_entity_density, self.rhetorical_template])
    def description(self): return ", ".join(k for k,v in asdict(self).items() if v) or "none"

class FingerprintMonitor:
    """
    Japanese-enhanced fingerprint monitor.
    CONN/ASRT sets extended with Japanese markers.
    Entity rate (er) threshold relaxed for Japanese text:
    Japanese tokens naturally contain kanji/katakana so the
    original er<0.05 threshold would rarely trigger.
    Adjusted to er<0.10 for Japanese-inclusive detection.
    """
    CONN = {
        "however","therefore","clearly","obviously","certainly",
        "moreover","furthermore","thus","hence","undoubtedly",
        # Japanese
        "しかし","したがって","つまり","要するに","さらに",
        "一方","ところで","明らかに","もちろん","当然",
    }
    ASRT = {
        "must","certainly","clearly","definitely","always",
        "never","absolutely","guaranteed","proven",
        # Japanese
        "絶対に","確かに","必須","確実","間違いなく","断言","明言","疑いなく",
    }
    def __init__(self, window_size=20): self.window_size = window_size

    def analyze(self, tokens):
        n = max(len(tokens), 1)
        low = [t.lower() for t in tokens]
        cr = sum(1 for t in low if t in self.CONN) / n
        ar = sum(1 for t in low if t in self.ASRT) / n
        # Japanese: kanji/katakana tokens also count as entity-like
        # Threshold relaxed to 0.10 to avoid over-triggering on Japanese text
        er = sum(1 for t in tokens if t and (
            t[0].isupper() or re.search(r'[一-龯ァ-ヴー]{2,}', t)
        )) / n
        return FingerprintSignal(cr>0.15, ar>0.10, er<0.10, cr>0.10 and ar>0.08)

    def pre_check(self, query): return self.analyze(query.split())

class RiskEngine:
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, seed_matcher=None, domain: str = "general"):
        if domain in DOMAIN_COEFFICIENTS:
            self.alpha, self.beta, self.gamma = DOMAIN_COEFFICIENTS[domain]
        else:
            self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.domain = domain
        self.eps = 1e-6
        self.seed = seed_matcher or SeedMatcher()
        self._dep_detector = ImplicitDependencyDetector()  # v11.5
        self._ht_w = HT_WEIGHTS.get(domain, (0.6, 0.2, 0.2))  # v11.5

    def calculate_lt(self, rf, gm):
        raw = self.alpha*(rf-gm) + self.beta*(rf/(gm+self.eps)) + self.gamma*(1-gm)
        return float(1.0/(1.0+math.exp(-raw)))

    def calculate_ht(self, text, context: str = ""):
        """
        v11.5: Strengthened Ht proxy.
        Combines TF-IDF seed matching with two deterministic
        implicit-dependency rules (future-reference, unresolved-variable).
        Weights are domain-calibrated via HT_WEIGHTS.
        """
        seed_score = float(max(0.0, min(1.0, 1.0 - self.seed.hit_rate(text))))
        future_flag, variable_flag = self._dep_detector.check(text, context)
        w1, w2, w3 = self._ht_w
        raw = w1 * seed_score + w2 * float(future_flag) + w3 * float(variable_flag)
        return float(max(0.0, min(1.0, raw)))

    def _rf(self, toks):
        """
        Japanese-enhanced rhetorical fluency.
        Adds Japanese connectives and assertiveness markers alongside English.
        Sentence splitting extended to Japanese punctuation (。！？).
        """
        CONNECTIVES = {
            # English
            "therefore","thus","hence","consequently","accordingly",
            "moreover","furthermore","additionally","besides","also",
            "however","nevertheless","nonetheless","yet","although",
            "despite","conversely","clearly","obviously","evidently",
            "undoubtedly","certainly","absolutely","definitely","indeed","truly","surely",
            # Japanese — discourse connectives
            "したがって","よって","それゆえ","ゆえに","つまり","すなわち","要するに",
            "しかし","だが","けれども","ただし","しかも","さらに","加えて",
            "一方","他方","逆に","ところで","さて","なお","また","および",
        }
        ASSERTIVENESS = {
            # English
            "must","always","never","impossible","guaranteed","definitive",
            "conclusive","prove","proven","fact","undeniable","irrefutable","unquestionable",
            # Japanese — assertiveness / overconfidence markers
            "明らかに","確かに","絶対に","必須","確実","間違いなく",
            "疑いなく","当然","断言","明言","証明","事実上","不可欠","決して",
        }
        t_lower = [t.lower() for t in toks]
        n = max(len(toks), 1)
        conn_rate = sum(1 for t in t_lower if t in CONNECTIVES) / n
        asrt_rate = sum(1 for t in t_lower if t in ASSERTIVENESS) / n
        # Japanese punctuation added to sentence splitter
        sentences = re.split(r'[.!?。！？]', " ".join(toks))
        lens = [len(s.split()) for s in sentences if s.strip()]
        template_signal = 0.0
        if len(lens) >= 2:
            mean_l = sum(lens) / len(lens)
            variance = sum((l - mean_l)**2 for l in lens) / len(lens)
            template_signal = max(0.0, 1.0 - min(1.0, variance / 20.0))
        return float(min(1.0, conn_rate * 3.0 + asrt_rate * 3.0 + template_signal * 0.2))

    def _gm(self, toks):
        """
        Japanese-enhanced grounding mass.

        Entity detection:
          English: leading uppercase, len > 1 (original logic)
          Japanese: kanji/katakana sequences of 2+ chars, excluding
                    common function words (は、が、を、に、で、と、も、の、へ、より).
                    Single-char kanji and hiragana-only tokens are excluded
                    to avoid inflating GM with grammatical particles and
                    verb endings — the main over-counting risk in naive
                    Japanese entity detection.

        Citation patterns:
          Added Japanese citation markers (によると, 参考, 出典, 文献,
          第N章, 図N, [N], （N）).

        Units:
          Added Japanese units (円, 万円, 億円, 年, 月, 日, ㎡, ℃, km², ha).
        """
        n = max(len(toks), 1)

        # Japanese function words to exclude from entity count
        JA_PARTICLES = {"は","が","を","に","で","と","も","の","へ","より","から","まで","など",
                        "です","ます","した","ある","いる","する","なる","れる","られる","ない"}

        def _is_ja_entity(t):
            # Katakana sequence of 2+ chars → likely loanword/proper noun
            if re.fullmatch(r'[ァ-ヴー]{2,}', t):
                return True
            # Kanji-containing token of 2+ chars, not a pure function word
            if len(t) >= 2 and re.search(r'[一-龯]', t) and t not in JA_PARTICLES:
                return True
            return False

        entity_count = sum(
            1 for i, t in enumerate(toks)
            if t and (
                (t[0].isupper() and len(t) > 1) or _is_ja_entity(t)
            ) and (i == 0 or toks[i-1][-1] not in '.!?。！？')
        )
        entity_density = entity_count / n

        # Citation patterns: English + Japanese
        citation_count = sum(
            1 for t in toks if (
                t.startswith("http") or
                t.startswith("[") or
                t.startswith("(") or
                (t.startswith('"') and len(t) > 3) or
                re.search(r'によると|参考|出典|文献|第[0-9０-９]+章|図[0-9０-９]+', t) or
                re.search(r'[\[（][0-9０-９]+[\]）]', t)
            )
        )
        citation_rate = citation_count / n

        # Units: English + Japanese
        UNITS = {"%","kg","km","ms","$","°","Hz","MB","GB","sec","min","hr","mph","°C","°F",
                 "円","万円","億円","年","月","日","㎡","℃","km²","ha","人","件","回"}
        unit_count = sum(1 for t in toks if any(u in t for u in UNITS))
        numeric_count = sum(1 for t in toks if re.search(r'\d', t))
        constraint_density = (unit_count + numeric_count * 0.3) / n

        return float(min(1.0, entity_density + citation_rate + constraint_density))

    def estimate(self, tokens, context: str = ""):
        text = " ".join(tokens)
        rf = self._rf(tokens)
        gm = self._gm(tokens)
        return RiskMetrics(self.calculate_ht(text, context), self.calculate_lt(rf, gm), rf, gm)

    def update_coefficients(self, da=0.0, db=0.0, dg=0.0):
        self.alpha = max(0.1, min(2.0, self.alpha + da))
        self.beta  = max(0.1, min(3.0, self.beta  + db))
        self.gamma = max(0.1, min(1.0, self.gamma + dg))

class DriftCalibrator:
    """
    EXPERIMENTAL / OPTIONAL component.
    Auto-nudges alpha/beta/gamma based on HOLD rate over a sliding window.
    Do NOT rely on this for reproducible reported results.
    For fixed-coefficient runs, instantiate RiskEngine with DOMAIN_COEFFICIENTS
    directly and skip calibrate() calls.
    """
    TARGET=0.20; NUDGE=0.02; WINDOW=50
    def __init__(self, engine): self.engine=engine; self._log=deque(maxlen=self.WINDOW)
    def record(self, at): self._log.append(at)
    def calibrate(self):
        if len(self._log)<10: return
        hold_types = (ActionType.HOLD, ActionType.SOFT_HOLD, ActionType.CLARIFY, ActionType.RETRIEVE)
        rate = sum(1 for a in self._log if a in hold_types)/len(self._log)
        delta = rate - self.TARGET
        if delta > 0.10: self.engine.update_coefficients(-self.NUDGE, -self.NUDGE, 0)
        elif delta < -0.10: self.engine.update_coefficients(+self.NUDGE, +self.NUDGE, 0)

class SoftHoldAnnotator:
    def __init__(self, engine, lt_warn=0.55): self.engine=engine; self.lt_warn=lt_warn
    def annotate(self, text):
        """
        v11.5: Pass full text as context to estimate() so that
        implicit-dependency rules (future-reference, unresolved-variable)
        can use the surrounding context for cancellation checks.
        """
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        out=[]; warns=[]
        for s in sents:
            # Pass full text as context — activates v11.5 Ht proxy cancellation
            m = self.engine.estimate(s.split(), context=text)
            if m.halation_risk > self.lt_warn:
                r=f"Lt={m.halation_risk:.2f},RF={m.rhetorical_fluency:.2f},GM={m.grounding_mass:.2f}"
                out.append(f"{s} [要検証: {r}]"); warns.append(r)
            else: out.append(s)
        return " ".join(out), warns

class RecoveryProtocol:
    @staticmethod
    def suggest(action, stakes=0.5):
        if stakes > 0.8: return "ESCALATE: Stakes too high. Consult domain expert."
        if action.partial_text and len(action.partial_text.split()) > 10:
            return "RESUME: Safe partial text available. Restart with added grounding."
        return "REPHRASE: Provide more specific context or cite sources."

@dataclass
class ObserverReport:
    name:str; risk_score:float; weight:float=1.0; note:str=""

class FactObserver:
    name="Ofact"; weight=1.0
    def evaluate(self, text, seed): return ObserverReport(self.name, 1.0-seed.hit_rate(text), self.weight)

class CohObserver:
    name="Ocoh"; weight=1.0
    def evaluate(self, text):
        lens=[len(s.split()) for s in text.split(".") if s.strip()]
        mu=sum(lens)/max(len(lens),1)
        var=sum((l-mu)**2 for l in lens)/max(len(lens),1)
        return ObserverReport(self.name, min(1.0,var/50), self.weight)

class SafeObserver:
    name="Osafe"; weight=2.0
    def evaluate(self, stakes, irrev):
        return ObserverReport(self.name, min(1.0, stakes*(1.5 if irrev else 1.0)), self.weight)

class VerbosityBiasObserver:
    name="Ovb"; weight=0.8
    def evaluate(self, rf): return ObserverReport(self.name, rf, self.weight)

class CreativeObserver:
    name="Ocre"; weight=0.5
    def evaluate(self, text):
        found=sum(1 for s in ["imagine","story","fiction","hypothetically"] if s in text.lower())
        return ObserverReport(self.name, max(0.0,1.0-found*0.3), self.weight)

class SafetyGating:
    def __init__(self, lam=5.0): self.lam=lam
    def aggregate(self, reports):
        ws=[r.weight for r in reports]; rs=[r.risk_score for r in reports]
        ew=[w*math.exp(self.lam*r) for w,r in zip(ws,rs)]
        tot=sum(ew) or 1.0
        return float(sum((e/tot)*r for e,r in zip(ew,rs)))

class Prism:
    """
    Japanese-enhanced deterministic post-check — NOT a strong verifier.
    Catches structural failures: empty output, truncation, unbalanced
    brackets (English and Japanese), missing citation markers on assertive
    claims (English and Japanese assertiveness markers).
    Intentionally lightweight; semantic validation requires external tools.
    Paper reference: Section 10 (Prism: deterministic post-check).
    """
    def validate(self, text):
        """
        Japanese-enhanced deterministic post-check — NOT a strong verifier.
        Adds Japanese truncation (…), bracket balance （）「」【】,
        citation patterns (によると/参考/出典/文献), and assertiveness
        markers (明らかに/確かに/絶対に/証明) alongside English heuristics.
        Paper reference: Section 10 (Prism: deterministic post-check).
        """
        failed = []
        if len(text.strip()) == 0:
            failed.append("non_empty")
        if text.strip().endswith(("...", "…")):
            failed.append("no_truncation")
        if text.count("(") != text.count(")"):
            failed.append("balanced_parens")
        if text.count("（") != text.count("）"):
            failed.append("balanced_ja_parens")
        if text.count("「") != text.count("」"):
            failed.append("balanced_ja_quotes")
        if text.count("【") != text.count("】"):
            failed.append("balanced_ja_brackets")
        has_citation = re.search(
            r'http[s]?://|\[|\d+年|\d+%|according to'
            r'|によると|参考|出典|文献|第[0-9]+章|図[0-9]+',
            text.lower()
        )
        assertive_en = any(w in text.lower() for w in ["must", "fact", "prove", "certainly"])
        assertive_ja = any(w in text for w in ["明らかに", "確かに", "絶対に", "証明", "間違いなく", "断言"])
        if not has_citation and (assertive_en or assertive_ja):
            failed.append("citation_missing")
        return (False, f"Prism failed: {failed}") if failed else (True, "OK")


@dataclass
class ReikaConfig:
    profile: BasinProfile = BasinProfile.DEEP_BASIN
    window_size: int = 20
    _ssde: SSDEParams = field(init=False)
    def __post_init__(self): self._ssde = BASIN_SSDE[self.profile]
    @property
    def ht_threshold(self): return self._ssde.ht_threshold
    @property
    def lt_threshold(self): return self._ssde.lt_threshold
    @property
    def obs_threshold(self): return self._ssde.obs_threshold
    @property
    def lambda_val(self): return self._ssde.lam
    def hash(self):
        raw=json.dumps({"profile":self.profile.value,"V":self._ssde.V,
                        "sigma":self._ssde.sigma,"lam":self._ssde.lam},
                       sort_keys=True).encode()
        return hashlib.md5(raw).hexdigest()[:8]

class ReikaController:
    def __init__(self, llm, config=None, dev_mode: bool = False,
                 enable_drift_calibration: bool = False):
        self.llm=llm; self.config=config or ReikaConfig()
        self.seed=SeedMatcher(); self.fp=FingerprintMonitor(self.config.window_size)
        domain_map = {
            BasinProfile.NARROW_BASIN: "medical_legal",
            BasinProfile.DEEP_BASIN: "technical",
            BasinProfile.WIDE_BASIN: "creative",
            BasinProfile.FAST_DECAY: "general",
        }
        self.risk=RiskEngine(seed_matcher=self.seed, domain=domain_map.get(self.config.profile, "general"))
        self.gating=SafetyGating(self.config.lambda_val); self.prism=Prism()
        self.annotator=SoftHoldAnnotator(self.risk)
        # DriftCalibrator is EXPERIMENTAL — disabled by default.
        # Enable explicitly via enable_drift_calibration=True for exploratory use only.
        # Do not rely on calibration results for reproducible reported results.
        self.calibrator=DriftCalibrator(self.risk) if enable_drift_calibration else None
        self.recovery=RecoveryProtocol()
        self.fact_obs=FactObserver(); self.coh_obs=CohObserver()
        self.safe_obs=SafeObserver(); self.vb_obs=VerbosityBiasObserver()
        self.cre_obs=CreativeObserver(); self.state="IDLE"
        self.term_halation=TerminologicalHalationDetector()
        self.kernel_guard=KernelExposureGuard(dev_mode=dev_mode)

    async def run(self, query, stakes=0.5, irreversible=False):
        self.state = "ACTIVE"; ch = self.config.hash()

        safe, reason = self.kernel_guard.check(query)
        if not safe:
            a = Action(ActionType.HOLD, reason, config_hash=ch,
                       recovery_hint="Remove kernel specification from input.")
            if self.calibrator: self.calibrator.record(a.type)
            return a

        fp=self.fp.pre_check(query)
        if fp.anomaly_detected:
            a=Action(ActionType.CLARIFY,f"Layer0 pre-check: {fp.description()}",config_hash=ch,
                     recovery_hint=self.recovery.suggest(Action(ActionType.CLARIFY,""),stakes))
            if self.calibrator: self.calibrator.record(a.type)
            return a

        self.state="GEN"; buf=[]; abort=None
        async for tok in self.llm.stream(query):
            buf.append(tok)
            if len(buf) % self.config.window_size == 0:
                win=buf[-self.config.window_size:]
                fp2=self.fp.analyze(win)
                # Streaming estimate is intentionally context-light for latency.
                # Strengthened Ht proxy (v11.5) is applied more fully at final evaluation.
                m=self.risk.estimate(win)
                if fp2.anomaly_detected:
                    abort=Action(ActionType.CLARIFY,f"Layer0 mid-stream: {fp2.description()}", partial_text=" ".join(buf),metrics=m.to_dict(),config_hash=ch); break
                if m.halation_risk>self.config.lt_threshold and m.hallucination_risk>self.config.ht_threshold:
                    abort=Action(ActionType.HOLD,"Dual high-risk during streaming", partial_text=" ".join(buf),metrics=m.to_dict(),config_hash=ch, recovery_hint=self.recovery.suggest(Action(ActionType.HOLD,"",partial_text=" ".join(buf)),stakes)); break
                if m.hallucination_risk>self.config.ht_threshold:
                    abort=Action(ActionType.RETRIEVE,f"Ht={m.hallucination_risk:.2f}", partial_text=" ".join(buf),metrics=m.to_dict(),config_hash=ch); break

        if abort:
            self.state=abort.type.value
            if self.calibrator: self.calibrator.record(abort.type)
            return abort

        full = " ".join(buf)
        m = self.risk.estimate(buf, context=query)  # v11.5: pass query as context

        term_score = self.term_halation.score(full)
        if term_score > 0.65:
            a = Action(ActionType.HOLD, f"Terminological Halation detected (score={term_score:.2f})",
                       partial_text=full[:100]+"...", metrics=m.to_dict(), config_hash=ch,
                       recovery_hint="Kernel-like vocabulary without grounding detected.")
            if self.calibrator: self.calibrator.record(a.type)
            return a

        reports=[self.fact_obs.evaluate(full,self.seed), self.coh_obs.evaluate(full),
                 self.safe_obs.evaluate(stakes,irreversible),
                 self.vb_obs.evaluate(m.rhetorical_fluency), self.cre_obs.evaluate(full)]
        agg=self.gating.aggregate(reports)
        obs={r.name: round(r.risk_score,3) for r in reports}
        ht,lt=m.hallucination_risk,m.halation_risk

        if ht>=self.config.ht_threshold and lt>=self.config.lt_threshold:
            a=Action(ActionType.HOLD,"Dual high-risk (Ht+Lt)",partial_text=full[:100]+"...", metrics=m.to_dict(),observer_scores=obs,config_hash=ch, recovery_hint=self.recovery.suggest(Action(ActionType.HOLD,"",partial_text=full),stakes))
        elif lt>=self.config.lt_threshold:
            ann,_=self.annotator.annotate(full)
            a=Action(ActionType.SOFT_HOLD,f"Lt={lt:.2f}: halation only → annotate", output_text=ann,metrics=m.to_dict(),observer_scores=obs,config_hash=ch, recovery_hint="Output annotated with [要検証] tags. Review before use.")
        elif ht>=self.config.ht_threshold:
            a=Action(ActionType.RETRIEVE,f"Ht={ht:.2f}: verification needed", partial_text=full,metrics=m.to_dict(),observer_scores=obs,config_hash=ch)
        elif agg>=self.config.obs_threshold:
            a=Action(ActionType.CLARIFY,f"Observer agg={agg:.2f}", partial_text=full,metrics=m.to_dict(),observer_scores=obs,config_hash=ch)
        else:
            ok,msg=self.prism.validate(full)
            a=Action(ActionType.OUTPUT if ok else ActionType.CLARIFY, "All checks passed" if ok else msg, output_text=full if ok else "", partial_text="" if ok else full, metrics=m.to_dict(),observer_scores=obs,config_hash=ch)

        self.state=a.type.value
        if self.calibrator:
            self.calibrator.record(a.type)
            self.calibrator.calibrate()
        return a

AVOIDED_COST={ActionType.HOLD:10.0,ActionType.SOFT_HOLD:4.0,ActionType.CLARIFY:5.0,ActionType.RETRIEVE:3.0,ActionType.OUTPUT:0.0}

def log_safety_metrics(sid, action):
    log=action.to_log(); log["session"]=sid
    log["roi_estimate"]=AVOIDED_COST.get(action.type,0.0); return log

class MockLLMGateway:
    # Scenario descriptions:
    #   normal      : well-grounded, low-risk output (expected: OUTPUT)
    #   halation    : high RF / low GM — fluent but ungrounded (expected: HOLD or SOFT_HOLD)
    #   hallucination: fabricated factual claims not in seed facts (expected: RETRIEVE)
    #   creative    : fictional framing, low grounding expected (expected: OUTPUT in WIDE_BASIN)
    SCENARIOS={
        "normal":
            "Based on the provided context the main finding suggests further investigation "
            "with reference to attached data from 2024 Q3 report section 4.2",
        "halation":
            "Clearly this must be true furthermore obviously it is absolutely certain that "
            "therefore we definitively conclude this is certainly correct and undeniable",
        "hallucination":
            "The boiling point of water is 87 degrees celsius and the capital of France "
            "is Lyon according to standard European geography textbooks published in 2019",
        "creative":
            "Imagine a story where hypothetically speaking let us say characters explore "
            "a fictional world in this narrative without any factual constraints",
        "future_reference":
            "Tomorrow the market will recover and next quarter earnings will improve "
            "within 3 months as the product launch drives growth by end of year",
        "unresolved_variable":
            "Based on the total amount the required specification should be met "
            "as mentioned above the relevant data confirms the appropriate threshold",
    }
    def __init__(self, scenario="normal"): self.scenario=scenario
    async def stream(self, query):
        for tok in self.SCENARIOS.get(self.scenario,"").split():
            await asyncio.sleep(0.002); yield tok

async def demo():
    print("ReikaSystem v11.4 Reference Implementation (with v11.5 patch set)")
    print("(arXiv prototype)")
    print("="*70)
    print("Basin profile guide:")
    print("  NARROW_BASIN : medical/legal — strict safety, low tolerance")
    print("  DEEP_BASIN   : finance/analysis — balanced, citation-sensitive")
    print("  FAST_DECAY   : UI/search — speed-first, shallow safety")
    print("  WIDE_BASIN   : creative — high RF tolerated, low GM expected")
    print("="*70)
    cases=[
        # expected: OUTPUT (well-grounded, low-risk)
        ("NARROW_BASIN + normal",      BasinProfile.NARROW_BASIN, "normal",        0.3, False),
        # expected: HOLD or CLARIFY (high RF, low GM — Layer 0 early exit likely)
        ("NARROW_BASIN + halation",    BasinProfile.NARROW_BASIN, "halation",      0.3, False),
        # expected: CLARIFY or SOFT_HOLD (wide basin tolerates higher RF)
        ("WIDE_BASIN   + halation",    BasinProfile.WIDE_BASIN,   "halation",      0.3, False),
        # expected: RETRIEVE or CLARIFY (fabricated facts, Ht elevated)
        ("DEEP_BASIN   + hallucination",BasinProfile.DEEP_BASIN,  "hallucination", 0.5, False),
        # expected: OUTPUT or CLARIFY (creative framing, low GM expected)
        ("WIDE_BASIN   + creative",    BasinProfile.WIDE_BASIN,   "creative",      0.2, False),
        # expected: HOLD (high stakes + halation = dual escalation)
        ("NARROW_BASIN + high stakes", BasinProfile.NARROW_BASIN, "halation",      0.9, True),
        # v11.5: expected: RETRIEVE (future-reference flag triggered)
        ("NARROW_BASIN + future_ref",  BasinProfile.NARROW_BASIN, "future_reference", 0.5, False),
        # v11.5: expected: RETRIEVE or CLARIFY (unresolved-variable flag triggered)
        ("NARROW_BASIN + unresolved",  BasinProfile.NARROW_BASIN, "unresolved_variable", 0.5, False),
    ]
    for label,profile,scenario,stakes,irrev in cases:
        cfg=ReikaConfig(profile=profile); llm=MockLLMGateway(scenario=scenario)
        ctrl=ReikaController(llm,cfg); a=await ctrl.run("Test",stakes=stakes,irreversible=irrev)
        log=log_safety_metrics(label,a); ssde=BASIN_SSDE[profile]
        print(f"\n{'─'*70}")
        print(f"Case    : {label}")
        print(f"Profile : V={ssde.V} σ={ssde.sigma} λ={ssde.lam} → Ht≤{cfg.ht_threshold:.2f} Lt≤{cfg.lt_threshold:.2f}")
        print(f"Action  : {log['action']:12s} | {log['reason']}")
        if a.output_text: print(f"Output  : {a.output_text[:70]}...")
        if a.recovery_hint: print(f"Recovery: {a.recovery_hint}")
        print(f"Metrics : {log['metrics']}")
        print(f"ROI     : {log['roi_estimate']}  Config#: {log['config_hash']}")

if __name__=="__main__":
    asyncio.run(demo())
