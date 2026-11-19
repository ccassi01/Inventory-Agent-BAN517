"""Supply Chain Analytics Inventory Decision Agent.

Implements the explicit 7-stage decision pipeline:
    1. Customer Question
    2. Identify Model
    3. Extract Inputs (whole-question semantic value + meaning extraction before checking missing fields)
    4. Select Python Tool
    5. Calculate (exclusively via inventory_tools.py)
    6. Verify (independent mathematical & sanity verification)
    7. Explain (business-friendly executive summary + formulas)

Never invents missing numerical inputs; examines the entire question for all
synonyms and contextual phrasing before deciding that any parameter is missing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from inventory_tools import TOOL_REGISTRY, verify_calculation

# ---------------------------------------------------------------------------
# Structured Pipeline Response
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Complete structured trace of all 7 pipeline stages."""

    question: str
    identified_model: str | None
    model_display_name: str | None
    identification_confidence: float
    identification_signals: list[str] = field(default_factory=list)
    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    extracted_inputs: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    selected_tool: str | None = None
    calculation_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    explanation: str = ""
    status: str = "ok"  # "ok", "needs_clarification", "error"
    error_message: str | None = None
    parser_mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper Parsing & Semantic Entity Utilities
# ---------------------------------------------------------------------------

_NUM_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUM_KM = rf"{_NUM_PATTERN}(?:[kKmM])?"

# Reusable ordering cost patterns covering all common EOQ / Quantity Discount synonyms
_ORDERING_COST_PATTERNS: list[str] = [
    rf"(?:ordering\s*/\s*setup|order\s*/\s*setup|setup\s*/\s*ordering|fixed\s+ordering|fixed\s+order|order\s+processing|ordering|order|setup)\s+(?:cost|expense|fee)s?\s*(?:is|are|of|equals?|=|:)?\s*\$?\s*({_NUM_PATTERN})",
    rf"(?:cost|expense)\s+(?:per\s+order|of\s+placing\s+(?:an|each|a\s+single)?\s*order|to\s+place\s+(?:an|each|a\s+single)?\s*order)\s*(?:is|of|equals?|=|:)?\s*\$?\s*({_NUM_PATTERN})",
    rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+order|/\s*order|(?:for|to)\s+place\s+(?:an|each)\s+order|(?:ordering\s*/\s*setup|ordering|order|setup)\s+(?:cost|expense)|cost\s+per\s+order)",
    rf"(?:each\s+order|placing\s+(?:an|each)\s+order|it)\s+(?:costs?|incurs?)\s*(?:a\s+)?(?:(?:ordering|order|setup)\s+(?:cost|expense)\s+of\s+)?\$\s*({_NUM_PATTERN})(?:\s*(?:per\s+order|to\s+place(?:\s+an\s+order)?))?",
    rf"incurs?\s+(?:a|an)\s+(?:ordering\s*/\s*setup|ordering|order|setup)\s+(?:cost|expense)\s+of\s*\$\s*({_NUM_PATTERN})",
    rf"\b(?:S|K)\s*=\s*\$?\s*({_NUM_PATTERN})",
]


def _parse_number(raw: str) -> float:
    cleaned = raw.strip().replace(",", "").replace("$", "")
    multiplier = 1.0
    if cleaned.lower().endswith("k"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1]
    elif cleaned.lower().endswith("m"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def _find_first_match(text: str, patterns: list[str]) -> tuple[float, str] | None:
    """Return (parsed_number, matched_text) for the first matching regex across the text."""
    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            try:
                val = _parse_number(match.group(1))
                return val, match.group(0).strip()
            except (ValueError, IndexError):
                continue
    return None


def _split_clauses(text: str) -> list[str]:
    """Split natural-language text into meaningful sentence/clause chunks without breaking numbers."""
    # Protect decimal points (e.g. 1.5) and commas in numbers (e.g. 5,000)
    raw_sentences = re.split(r"(?<!\d)\.(?!\d)|[;\n]+", text)
    clauses: list[str] = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        clauses.append(sent)
    return clauses


def _detect_time_unit(segment: str, default: str = "day") -> str:
    s = segment.lower()
    if any(w in s for w in ("day", "daily")):
        return "day"
    if any(w in s for w in ("week", "weekly")):
        return "week"
    if any(w in s for w in ("month", "monthly")):
        return "month"
    if any(w in s for w in ("year", "annual", "annually", "yearly")):
        return "year"
    return default


def _convert_time_value(value: float, from_unit: str, to_unit: str) -> tuple[float, str | None]:
    if from_unit == to_unit:
        return value, None
    days_map = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
    if from_unit in days_map and to_unit in days_map:
        converted = value * (days_map[from_unit] / days_map[to_unit])
        note = f"Converted {value:g} {from_unit}(s) to {converted:g} {to_unit}(s) to align time units."
        return converted, note
    return value, None


def _record_entity(
    entities: list[dict[str, Any]],
    parameter: str,
    meaning: str,
    value: Any,
    matched_text: str,
    unit: str | None = None,
) -> None:
    entities.append(
        {
            "parameter": parameter,
            "meaning": meaning,
            "value": value,
            "unit": unit,
            "matched_text": matched_text,
        }
    )


def _extract_price_breaks(text: str) -> tuple[list[dict[str, Any]], float | None, list[dict[str, Any]]]:
    """Extract all-units quantity discount price breaks and any base unit purchase price.

    Returns:
        (tiers, base_unit_purchase_price, entity_records)
    """
    tiers: list[dict[str, Any]] = []
    seen_mins: set[float] = set()
    entities: list[dict[str, Any]] = []

    base_price_match = _find_first_match(
        text,
        [
            rf"(?:base|regular|standard|normal|list|unit)\s+purchase\s+price\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
            rf"(?:base|regular|standard|normal|list)\s+(?:unit\s+)?(?:price|cost)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
            rf"unit\s+purchase\s+(?:price|cost)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
        ],
    )
    base_price: float | None = None
    if base_price_match is not None:
        base_price, matched = base_price_match
        _record_entity(entities, "unit_purchase_price", "unit purchase price", base_price, matched, "$/unit")

    # Pattern A: Range "1-499 units at $20" or "500 to 999: $18.50" or "1,000–2,499 cost $16"
    range_pattern = re.compile(
        rf"\b({_NUM_PATTERN})\s*(?:-|–|to)\s*({_NUM_PATTERN})\s*(?:units?|items?|pcs?)?\s*(?:at|for|:|=|cost(?:s|ing)?|price(?:d)?\s*(?:at)?|@)?\s*\$\s*({_NUM_PATTERN})",
        flags=re.IGNORECASE,
    )
    for m in range_pattern.finditer(text):
        min_q = _parse_number(m.group(1))
        max_q = _parse_number(m.group(2))
        price = _parse_number(m.group(3))
        if min_q not in seen_mins:
            seen_mins.add(min_q)
            tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
            _record_entity(
                entities,
                "price_break",
                f"quantity break ({min_q:g}-{max_q:g} units) with unit price",
                {"min_qty": min_q, "max_qty": max_q, "unit_price": price},
                m.group(0).strip(),
                "$/unit",
            )

    # Pattern B: "under 500 at $20" / "less than 500 units: $20" / "up to 499 units at $20"
    under_pattern = re.compile(
        rf"\b(?:under|less\s+than|below|fewer\s+than|<)\s*({_NUM_PATTERN})\s*(?:units?|items?)?\s*(?:at|for|:|=|cost(?:s|ing)?|price(?:d)?\s*(?:at)?|@)?\s*\$\s*({_NUM_PATTERN})",
        flags=re.IGNORECASE,
    )
    for m in under_pattern.finditer(text):
        upper = _parse_number(m.group(1))
        price = _parse_number(m.group(2))
        min_q = 1.0
        max_q = upper - 1.0 if upper >= 2 else upper
        if min_q not in seen_mins:
            seen_mins.add(min_q)
            tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
            _record_entity(
                entities,
                "price_break",
                f"quantity break (<{upper:g} units) with unit price",
                {"min_qty": min_q, "max_qty": max_q, "unit_price": price},
                m.group(0).strip(),
                "$/unit",
            )

    upto_pattern = re.compile(
        rf"\bup\s+to\s*({_NUM_PATTERN})\s*(?:units?|items?)?\s*(?:at|for|:|=|cost(?:s|ing)?|price(?:d)?\s*(?:at)?|@)?\s*\$\s*({_NUM_PATTERN})",
        flags=re.IGNORECASE,
    )
    for m in upto_pattern.finditer(text):
        max_q = _parse_number(m.group(1))
        price = _parse_number(m.group(2))
        min_q = 1.0
        if min_q not in seen_mins:
            seen_mins.add(min_q)
            tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
            _record_entity(
                entities,
                "price_break",
                f"quantity break (up to {max_q:g} units) with unit price",
                {"min_qty": min_q, "max_qty": max_q, "unit_price": price},
                m.group(0).strip(),
                "$/unit",
            )

    # Pattern C: Open-ended "1000+ units at $16" / "1000 or more: $16" / ">= 1000 at $16" / "at least 1000: $16"
    plus_pattern = re.compile(
        rf"(?:(?:>=|at\s+least|over|more\s+than)\s*({_NUM_PATTERN})|({_NUM_PATTERN})\s*(?:\+|or\s+more|and\s+above|and\s+up))\s*(?:units?|items?|pcs?)?\s*(?:at|for|:|=|cost(?:s|ing)?|price(?:d)?\s*(?:at)?|@)?\s*\$\s*({_NUM_PATTERN})",
        flags=re.IGNORECASE,
    )
    for m in plus_pattern.finditer(text):
        raw_min = m.group(1) or m.group(2)
        min_q = _parse_number(raw_min)
        price = _parse_number(m.group(3))
        if min_q not in seen_mins:
            seen_mins.add(min_q)
            tiers.append({"min_qty": min_q, "max_qty": None, "unit_price": price})
            _record_entity(
                entities,
                "price_break",
                f"quantity break ({min_q:g}+ units) with unit price",
                {"min_qty": min_q, "max_qty": None, "unit_price": price},
                m.group(0).strip(),
                "$/unit",
            )

    # Pattern D: Reversed phrasing "$20 per unit for 1-499 units", "$16 for orders of 1000 or more"
    if not tiers:
        rev_range = re.compile(
            rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*|each\s*)?(?:for|if\s+ordering|on)\s*(?:orders?\s+of\s+)?({_NUM_PATTERN})\s*(?:-|–|to)\s*({_NUM_PATTERN})",
            flags=re.IGNORECASE,
        )
        for m in rev_range.finditer(text):
            price = _parse_number(m.group(1))
            min_q = _parse_number(m.group(2))
            max_q = _parse_number(m.group(3))
            if min_q not in seen_mins:
                seen_mins.add(min_q)
                tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
                _record_entity(
                    entities,
                    "price_break",
                    f"quantity break ({min_q:g}-{max_q:g} units) with unit price",
                    {"min_qty": min_q, "max_qty": max_q, "unit_price": price},
                    m.group(0).strip(),
                    "$/unit",
                )

        rev_under = re.compile(
            rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*|each\s*)?(?:for|if\s+ordering|on)\s*(?:orders?\s+of\s+)?(?:under|less\s+than|below|fewer\s+than|<)\s*({_NUM_PATTERN})",
            flags=re.IGNORECASE,
        )
        for m in rev_under.finditer(text):
            price = _parse_number(m.group(1))
            upper = _parse_number(m.group(2))
            min_q = 1.0
            max_q = upper - 1.0 if upper >= 2 else upper
            if min_q not in seen_mins:
                seen_mins.add(min_q)
                tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
                _record_entity(
                    entities,
                    "price_break",
                    f"quantity break (<{upper:g} units) with unit price",
                    {"min_qty": min_q, "max_qty": max_q, "unit_price": price},
                    m.group(0).strip(),
                    "$/unit",
                )

        rev_plus = re.compile(
            rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*|each\s*)?(?:for|if\s+ordering|on)\s*(?:orders?\s+of\s+)?(?:(?:>=|at\s+least)\s*({_NUM_PATTERN})|({_NUM_PATTERN})\s*(?:\+|or\s+more))",
            flags=re.IGNORECASE,
        )
        for m in rev_plus.finditer(text):
            price = _parse_number(m.group(1))
            min_q = _parse_number(m.group(2) or m.group(3))
            if min_q not in seen_mins:
                seen_mins.add(min_q)
                tiers.append({"min_qty": min_q, "max_qty": None, "unit_price": price})
                _record_entity(
                    entities,
                    "price_break",
                    f"quantity break ({min_q:g}+ units) with unit price",
                    {"min_qty": min_q, "max_qty": None, "unit_price": price},
                    m.group(0).strip(),
                    "$/unit",
                )

    # Pattern E: Percentage discount off a base unit purchase price
    if not tiers:
        if base_price is None:
            fallback_bp = _find_first_match(
                text,
                [
                    rf"(?:unit\s+price|unit\s+cost|purchase\s+price|item\s+price)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
                    rf"costs?\s*\$\s*({_NUM_PATTERN})\s*(?:per\s+unit|each)",
                ],
            )
            if fallback_bp is not None:
                base_price, matched = fallback_bp
                _record_entity(entities, "unit_purchase_price", "unit purchase price", base_price, matched, "$/unit")

        if base_price is not None:
            disc_pattern = re.compile(
                rf"({_NUM_PATTERN})\s*%\s*(?:discount|off)\s*(?:for|on|when\s+ordering|at)?\s*(?:orders?\s+of\s+)?(?:(?:>=|at\s+least)\s*)?({_NUM_PATTERN})\s*(?:\+|or\s+more|units?)?",
                flags=re.IGNORECASE,
            )
            discounts = [
                (float(m.group(1)), _parse_number(m.group(2)), m.group(0).strip())
                for m in disc_pattern.finditer(text)
            ]
            if discounts:
                discounts.sort(key=lambda x: x[1])
                tiers.append({"min_qty": 1.0, "max_qty": discounts[0][1] - 1.0, "unit_price": base_price})
                for i, (pct, min_q, matched) in enumerate(discounts):
                    next_max = discounts[i + 1][1] - 1.0 if i + 1 < len(discounts) else None
                    disc_price = round(base_price * (1.0 - pct / 100.0), 6)
                    tiers.append({"min_qty": min_q, "max_qty": next_max, "unit_price": disc_price})
                    _record_entity(
                        entities,
                        "price_break",
                        f"quantity break ({min_q:g}+ units, {pct:g}% discount off ${base_price:g})",
                        {"min_qty": min_q, "max_qty": next_max, "unit_price": disc_price},
                        matched,
                        "$/unit",
                    )

    # If a single high break was given alongside a base unit purchase price (e.g., "normally $20/unit, or $18 for 500+ units")
    if len(tiers) == 1 and base_price is not None and tiers[0]["min_qty"] > 1:
        first_break = tiers[0]["min_qty"]
        tiers.insert(0, {"min_qty": 1.0, "max_qty": first_break - 1.0, "unit_price": base_price})

    tiers.sort(key=lambda t: t["min_qty"])
    return tiers, base_price, entities


def _extract_discrete_scenarios(text: str) -> list[tuple[float, float]]:
    """Parse discrete (demand, probability) pairs from natural language."""
    scenarios: list[tuple[float, float]] = []
    # Pattern 1: "100 with probability 0.2" / "demand of 100 (prob 0.2)" / "100: 0.2"
    p1 = re.compile(
        rf"(?:demand\s+(?:of\s+)?)?({_NUM_PATTERN})\s*(?:units?\s*)?(?:with\s+(?:a\s+)?(?:probability|prob\.?|chance)\s*(?:of\s*)?|\(\s*(?:p\s*=\s*|prob\.?\s*)?)(0?\.\d+|{_NUM_PATTERN}\s*%)\s*\)?",
        flags=re.IGNORECASE,
    )
    for m in p1.finditer(text):
        d_val = _parse_number(m.group(1))
        raw_p = m.group(2).strip()
        prob = _parse_number(raw_p[:-1]) / 100.0 if raw_p.endswith("%") else float(raw_p)
        scenarios.append((d_val, prob))
    return scenarios


# ---------------------------------------------------------------------------
# Decision Agent Implementation
# ---------------------------------------------------------------------------


class InventoryDecisionAgent:
    """Supply Chain Analytics Inventory Decision Agent."""

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o-mini") -> None:
        self.use_llm = use_llm and bool(os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model

    # ------------------------------------------------------------------
    # Stage 2: Identify Model
    # ------------------------------------------------------------------

    def identify_model(self, question: str) -> tuple[str | None, float, list[str]]:
        """Identify which of the 5 inventory models matches the customer question."""
        q = question.lower()
        scores: dict[str, float] = {k: 0.0 for k in TOOL_REGISTRY}
        signals: dict[str, list[str]] = {k: [] for k in TOOL_REGISTRY}

        keyword_rules: dict[str, list[tuple[str, float, str]]] = {
            "newsvendor": [
                (r"\bnewsvendor\b|\bnewsboy\b|\bsingle[- ]period\b", 5.5, "explicit newsvendor / single-period"),
                (r"\bsalvage\b|\bmarkdown\b|\bclearance\b|\bliquidation\b|\bunsold\b|\bleftover\b", 3.5, "salvage / unsold value"),
                (r"\bunderage\b|\boverage\b|\bshortage\s+cost\b|\bexcess\s+cost\b|\bcritical\s+ratio\b", 4.5, "underage/overage or critical ratio"),
                (r"\bperishable\b|\bseasonal\b|\bholiday\b|\bone[- ]time\s+order\b|\bnewspaper\b|\bbakery\b|\bfashion\b", 2.5, "perishable / seasonal context"),
                (r"\bselling\s+price\b|\bretail\s+price\b|\bsale\s+price\b|\bsell(?:s|ing)?\s+(?:for|at)\s*\$", 2.0, "selling price present"),
            ],
            "quantity_discount": [
                (r"\bquantity\s+discount\b|\bprice\s+break\b|\bquantity\s+break\b|\bvolume\s+discount\b|\bbulk\s+discount\b|\ball[- ]units\b", 5.5, "explicit quantity discount / quantity break"),
                (r"\bdiscount\s+schedule\b|\bpricing\s+tier\b|\btiered\s+pric", 4.5, "tiered pricing schedule"),
                (r"\d+\s*(?:-|–|to)\s*\d+\s*(?:units?)?\s*(?:at|for|:|@)?\s*\$", 4.0, "quantity-price range detected"),
                (r"\d+\s*(?:\+|or\s+more)\s*(?:units?)?\s*(?:at|for|:|@)?\s*\$", 3.5, "threshold price break detected"),
                (r"\d+\s*%\s*(?:discount|off)\s*(?:for|on|when|at)\s*(?:orders?\s+of\s+)?\d+", 4.0, "percentage quantity discount detected"),
            ],
            "fixed_period_review": [
                (r"\bfixed[- ]period\b|\bperiodic\s+review\b|\border[- ]up[- ]to\b|\btarget\s+inventory\b", 5.5, "explicit fixed-period / target inventory"),
                (r"\breview\s+(?:period|interval|cycle)\b|\breview(?:ed|s|ing)?(?:\s+(?:inventory|stock))?\s+every\b|\bcheck(?:ed|s|ing)?(?:\s+(?:inventory|stock))?\s+every\b|\border(?:s|ed|ing)?\s+(?:are\s+placed\s+)?every\s+\d+", 4.5, "periodic review interval"),
                (r"\binventory\s+position\b|\bon[- ]hand\b|\bon[- ]order\b|\bbackorders?\b|\bscheduled\s+receipt", 2.0, "inventory position / on-hand / on-order / backorders"),
            ],
            "safety_stock_rop": [
                (r"\bsafety\s+stock\b|\breorder\s+point\b|\brop\b|\bcontinuous\s+review\b", 5.0, "explicit safety stock / reorder point"),
                (r"\bwhen\s+should\s+(?:we|i)\s+reorder\b|\bwhat\s+level\s+should\s+(?:we|i)\s+reorder\b", 4.0, "reorder trigger question"),
                (r"\blead[- ]time\s+standard\s+deviation\b|\blead\s+time\b", 1.5, "lead time mentioned"),
                (r"\bservice\s+level\b|\bcycle\s+service\b|\bstockout\b|\bz[- ]score\b", 1.5, "service level / stockout risk"),
            ],
            "eoq": [
                (r"\beoq\b|\beconomic\s+order\s+quantity\b", 5.5, "explicit EOQ"),
                (r"\border(?:ing)?\s*(?:/\s*setup)?\s+cost\b|\bsetup\s+(?:cost|expense)\b|\bcost\s+(?:per\s+order|of\s+placing\s+an\s+order)\b", 2.5, "ordering/setup cost"),
                (r"\bholding\s+cost\b|\bcarrying\s+cost\b", 2.0, "holding/carrying cost"),
                (r"\bannual\s+demand\b|\bdemand\s+per\s+year\b|\bannual\s+usage\b", 1.5, "annual demand"),
                (r"\boptimal\s+order\s+quantity\b|\bhow\s+many\s+units\s+should\s+(?:we|i)\s+order\b", 1.5, "order quantity optimization"),
            ],
        }

        for model_key, rules in keyword_rules.items():
            for pattern, weight, label in rules:
                if re.search(pattern, q):
                    scores[model_key] += weight
                    signals[model_key].append(label)

        tiers, _, _ = _extract_price_breaks(question)
        if len(tiers) >= 2:
            scores["quantity_discount"] += 4.5
            signals["quantity_discount"].append("multiple price-break tiers parsed")

        if re.search(
            r"\breview\s+(?:period|interval|cycle)\b|\breview(?:ed|s|ing)?(?:\s+(?:inventory|stock))?\s+every\b|\border[- ]up[- ]to\b|\bperiodic\b",
            q,
        ):
            scores["fixed_period_review"] += 3.0

        best_model = max(scores, key=lambda k: scores[k])
        best_score = scores[best_model]

        if best_score < 2.0:
            return None, 0.0, ["No strong model keywords matched."]

        total_positive = sum(v for v in scores.values() if v > 0)
        confidence = round(
            min(0.99, best_score / max(total_positive, 1.0) * 0.65 + min(best_score / 10.0, 0.35)),
            2,
        )
        return best_model, confidence, signals[best_model]

    # ------------------------------------------------------------------
    # Stage 3: Extract Inputs (Whole-Question Semantic Extraction)
    # ------------------------------------------------------------------

    def extract_inputs(
        self, question: str, model: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        """Extract numerical values and semantic meanings across the entire question
        BEFORE checking whether any required input is missing.

        Returns:
            (inputs, assumptions, missing_inputs, extracted_entities)
        """
        if self.use_llm:
            llm_result = self._extract_inputs_with_llm(question, model)
            if llm_result is not None:
                return llm_result

        if model == "eoq":
            return self._extract_eoq_inputs(question)
        if model == "quantity_discount":
            return self._extract_quantity_discount_inputs(question)
        if model == "safety_stock_rop":
            return self._extract_rop_inputs(question)
        if model == "fixed_period_review":
            return self._extract_periodic_review_inputs(question)
        if model == "newsvendor":
            return self._extract_newsvendor_inputs(question)
        raise ValueError(f"Unsupported model: {model}")

    # --- 1. EOQ Semantic Extractor ---

    def _extract_eoq_inputs(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        inputs: dict[str, Any] = {}
        assumptions: list[str] = []
        entities: list[dict[str, Any]] = []

        # Pass 1: Extract every recognized value & meaning across the entire question
        work_days_match = _find_first_match(
            q,
            [
                rf"({_NUM_PATTERN})\s*(?:working|operating|business)\s+days(?:\s*(?:per|a|in\s+a)\s+year)?",
                rf"(?:working|operating|business)\s+days(?:\s+per\s+year)?\s*(?:is|are|of|=|:)?\s*({_NUM_PATTERN})",
            ],
        )
        work_days_explicit: float | None = None
        if work_days_match is not None:
            work_days_explicit, matched = work_days_match
            inputs["working_days_per_year"] = work_days_explicit
            _record_entity(entities, "working_days_per_year", "working days per year", work_days_explicit, matched, "days/year")

        annual_match = _find_first_match(
            q,
            [
                rf"(?:annual|yearly)\s+(?:demand|usage|sales|consumption|requirement)\s*(?:is|of|equals?|=|:)?\s*({_NUM_KM})",
                rf"(?:demand|usage|sales|consumption)\s*(?:is|of|averages?|=|:)?\s*({_NUM_KM})\s*(?:units?\s*)?(?:per\s+year|a\s+year|annually|/year|/yr)",
                rf"(?:uses?|sells?|needs?|requires?|buys?|consumes?|demands?)\s*(?:an\s+average\s+of\s+)?({_NUM_KM})\s*units?\s*(?:per\s+year|a\s+year|annually|each\s+year)",
                rf"({_NUM_KM})\s*units?\s*(?:per\s+year|a\s+year|annually|/year|/yr)",
                rf"\bD\s*=\s*({_NUM_KM})",
            ],
        )
        if annual_match is not None:
            annual_d, matched = annual_match
            inputs["annual_demand"] = annual_d
            _record_entity(entities, "annual_demand", "annual demand", annual_d, matched, "units/year")
        else:
            monthly_match = _find_first_match(
                q,
                [
                    rf"monthly\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})\s*(?:units?\s*)?(?:per\s+month|monthly|/month)",
                    rf"({_NUM_KM})\s*units?\s*(?:per\s+month|a\s+month|monthly)",
                ],
            )
            weekly_match = _find_first_match(
                q,
                [
                    rf"weekly\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})\s*(?:units?\s*)?(?:per\s+week|weekly|/week)",
                    rf"({_NUM_KM})\s*units?\s*(?:per\s+week|a\s+week|weekly)",
                ],
            )
            daily_match = _find_first_match(
                q,
                [
                    rf"daily\s+(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})",
                    rf"(?:demand|usage|sales)\s*(?:is|of|=|:)?\s*({_NUM_KM})\s*(?:units?\s*)?(?:per\s+day|daily|/day)",
                    rf"({_NUM_KM})\s*units?\s*(?:per\s+day|a\s+day|daily)",
                ],
            )
            if monthly_match is not None:
                m_val, matched = monthly_match
                annual_d = m_val * 12.0
                inputs["annual_demand"] = annual_d
                _record_entity(entities, "annual_demand", "monthly demand (annualized ×12)", annual_d, matched, "units/year")
                assumptions.append(f"Annualized monthly demand: {m_val:g} × 12 = {annual_d:g} units/year.")
            elif weekly_match is not None:
                w_val, matched = weekly_match
                annual_d = w_val * 52.0
                inputs["annual_demand"] = annual_d
                _record_entity(entities, "annual_demand", "weekly demand (annualized ×52)", annual_d, matched, "units/year")
                assumptions.append(f"Annualized weekly demand: {w_val:g} × 52 = {annual_d:g} units/year.")
            elif daily_match is not None and work_days_explicit is not None:
                d_val, matched = daily_match
                annual_d = d_val * work_days_explicit
                inputs["annual_demand"] = annual_d
                _record_entity(
                    entities,
                    "annual_demand",
                    f"daily demand (annualized ×{work_days_explicit:g} working days)",
                    annual_d,
                    matched,
                    "units/year",
                )
                assumptions.append(
                    f"Annualized daily demand: {d_val:g} × {work_days_explicit:g} working days = {annual_d:g} units/year."
                )

        # Ordering cost (all EOQ synonyms)
        ordering_match = _find_first_match(q, _ORDERING_COST_PATTERNS)
        if ordering_match is not None:
            ord_val, matched = ordering_match
            inputs["ordering_cost"] = ord_val
            _record_entity(entities, "ordering_cost", "ordering / setup cost per order", ord_val, matched, "$/order")

        # Unit purchase cost/price
        unit_cost_match = _find_first_match(
            q,
            [
                rf"(?:unit\s+purchase|purchase|unit|item)\s+(?:cost|price)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"(?:costs?|priced\s+at|bought\s+(?:at|for))\s*\$\s*({_NUM_PATTERN})\s*(?:per\s+unit|each|/\s*unit)",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit|each)\s*(?:purchase|unit)\s*(?:cost|price)",
                rf"\$\s*({_NUM_PATTERN})\s+unit\s+(?:purchase\s+)?(?:cost|price)",
                rf"\bC\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )
        if unit_cost_match is not None:
            uc_val, matched = unit_cost_match
            inputs["unit_cost"] = uc_val
            _record_entity(entities, "unit_cost", "unit purchase price / cost", uc_val, matched, "$/unit")

        # Holding cost: fixed ($) vs percentage (%)
        holding_rate_match = _find_first_match(
            q,
            [
                rf"(?:annual\s+)?(?:percentage\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s+(?:cost\s+)?(?:rate|charge|percentage)?\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*%",
                rf"({_NUM_PATTERN})\s*%\s*(?:annual\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s*(?:cost|rate|charge)?",
                rf"({_NUM_PATTERN})\s*%\s*of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:cost|price|value)",
                rf"(?:holding|carrying)\s+(?:cost\s+)?rate\s*(?:is|of|=|:)?\s*(0\.\d+)",
                rf"\b(?:I|h)\s*=\s*({_NUM_PATTERN})\s*%",
                rf"\b(?:I|h)\s*=\s*(0\.\d+)",
            ],
        )
        holding_cost_match = _find_first_match(
            q,
            [
                rf"(?:annual\s+)?(?:fixed\s+|constant\s+)?(?:inventory\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s+(?:cost|expense)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*(?:per\s+year|a\s+year|annually|/year|/yr)?|/\s*unit\s*/\s*(?:year|yr))\s*(?:to\s+(?:hold|carry|store)|(?:annual\s+)?(?:holding|carrying|storage)\s+cost)",
                rf"(?:hold(?:ing)?|carry(?:ing)?|stor(?:ing|e))\s+(?:a|one|each)?\s*unit\s*(?:for\s+a\s+year|per\s+year|annually)?\s*(?:is|costs?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"costs?\s*\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*(?:per\s+year|annually|/year|/yr))\s*(?:to\s+(?:hold|carry|store))?",
                rf"\bH\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )

        if holding_cost_match is not None:
            hc_val, matched = holding_cost_match
            inputs["holding_cost"] = hc_val
            _record_entity(entities, "holding_cost", "fixed annual holding cost", hc_val, matched, "$/unit/year")
        elif holding_rate_match is not None:
            hr_raw, matched = holding_rate_match
            hr_val = hr_raw / 100.0 if hr_raw > 1.0 else hr_raw
            inputs["holding_cost_rate"] = hr_val
            _record_entity(entities, "holding_cost_rate", "percentage holding cost rate", hr_val, matched, "rate/year")

        # Pass 2: Only check missing inputs after examining the entire question
        missing: list[str] = []
        if "annual_demand" not in inputs:
            missing.append("annual_demand (units/year)")
        if "ordering_cost" not in inputs:
            missing.append("ordering_cost ($/order)")
        if "holding_cost" not in inputs and not ("holding_cost_rate" in inputs and "unit_cost" in inputs):
            if "holding_cost_rate" in inputs and "unit_cost" not in inputs:
                missing.append("unit_cost ($/unit, needed to compute H = holding_cost_rate × unit_cost)")
            else:
                missing.append("holding_cost ($/unit/year) or both unit_cost ($) and holding_cost_rate (%)")

        return inputs, assumptions, missing, entities

    # --- 2. Quantity Discount Semantic Extractor ---

    def _extract_quantity_discount_inputs(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        inputs: dict[str, Any] = {}
        assumptions: list[str] = []
        entities: list[dict[str, Any]] = []

        # Annual demand
        annual_match = _find_first_match(
            q,
            [
                rf"(?:annual|yearly)\s+(?:demand|usage|sales|consumption|requirement)\s*(?:is|of|equals?|=|:)?\s*({_NUM_KM})",
                rf"(?:demand|usage|sales)\s*(?:is|of|averages?|=|:)?\s*({_NUM_KM})\s*(?:units?\s*)?(?:per\s+year|a\s+year|annually|/year|/yr)",
                rf"(?:uses?|sells?|needs?|requires?|buys?|consumes?|demands?)\s*({_NUM_KM})\s*units?\s*(?:per\s+year|a\s+year|annually|each\s+year)",
                rf"({_NUM_KM})\s*units?\s*(?:per\s+year|a\s+year|annually|/year|/yr)",
                rf"\bD\s*=\s*({_NUM_KM})",
            ],
        )
        if annual_match is not None:
            annual_d, matched = annual_match
            inputs["annual_demand"] = annual_d
            _record_entity(entities, "annual_demand", "annual demand", annual_d, matched, "units/year")

        # Ordering cost (all synonyms)
        ordering_match = _find_first_match(q, _ORDERING_COST_PATTERNS)
        if ordering_match is not None:
            ord_val, matched = ordering_match
            inputs["ordering_cost"] = ord_val
            _record_entity(entities, "ordering_cost", "ordering / setup cost per order", ord_val, matched, "$/order")

        # Distinguish percentage holding cost vs fixed holding cost (avoid matching "% discount")
        holding_rate_match = _find_first_match(
            q,
            [
                rf"(?:annual\s+)?(?:percentage\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s+(?:cost\s+)?(?:rate|charge|percentage)?\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*%",
                rf"({_NUM_PATTERN})\s*%\s*(?:annual\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s*(?:cost|rate|charge)?",
                rf"({_NUM_PATTERN})\s*%\s*of\s+(?:the\s+)?(?:unit\s+)?(?:purchase\s+)?(?:cost|price|value)",
                rf"(?:holding|carrying)\s+(?:cost\s+)?rate\s*(?:is|of|=|:)?\s*(0\.\d+)",
                rf"\b(?:I|h)\s*=\s*({_NUM_PATTERN})\s*%",
                rf"\b(?:I|h)\s*=\s*(0\.\d+)",
            ],
        )
        holding_cost_match = _find_first_match(
            q,
            [
                rf"(?:annual\s+)?(?:fixed\s+|constant\s+)?(?:inventory\s+)?(?:holding\s*/\s*carrying|holding|carrying|storage)\s+(?:cost|expense)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*(?:per\s+year|annually|/year|/yr)?)\s*(?:fixed\s+)?(?:holding|carrying|storage)\s+cost",
                rf"\bH\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )

        if holding_rate_match is not None:
            hr_raw, matched = holding_rate_match
            hr_val = hr_raw / 100.0 if hr_raw > 1.0 else hr_raw
            inputs["holding_cost_rate"] = hr_val
            _record_entity(entities, "holding_cost_rate", "percentage holding cost rate", hr_val, matched, "rate/year")
        elif holding_cost_match is not None:
            hc_val, matched = holding_cost_match
            inputs["holding_cost"] = hc_val
            _record_entity(entities, "holding_cost", "fixed annual holding cost", hc_val, matched, "$/unit/year")

        # Price breaks & base unit purchase price
        tiers, _, break_entities = _extract_price_breaks(q)
        entities.extend(break_entities)
        if tiers:
            inputs["price_breaks"] = tiers

        missing: list[str] = []
        if "annual_demand" not in inputs:
            missing.append("annual_demand (units/year)")
        if "ordering_cost" not in inputs:
            missing.append("ordering_cost ($/order)")
        if "holding_cost_rate" not in inputs and "holding_cost" not in inputs:
            missing.append("percentage holding_cost_rate (% of unit price) or fixed holding_cost ($/unit/year)")
        if "price_breaks" not in inputs:
            missing.append("price_breaks (quantity breaks and unit prices)")

        return inputs, assumptions, missing, entities

    # --- Shared Clause-Aware Extractor for ROP & Fixed-Period Review ---

    def _extract_demand_leadtime_service(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], str, list[dict[str, Any]]]:
        """Clause-aware semantic extractor distinguishing:
            - average demand
            - demand standard deviation
            - average lead time
            - lead-time standard deviation
            - service level / z-score
        """
        inputs: dict[str, Any] = {}
        assumptions: list[str] = []
        entities: list[dict[str, Any]] = []
        clauses = _split_clauses(q)

        demand_period = "day"

        # 1. Extract average demand (and detect demand time unit)
        # First check compact N(mu, sigma) demand notation
        demand_normal_match = re.search(
            rf"demand\s*(?:is|~|follows)?\s*N\(\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*\)\s*(?:units?\s*)?(?:per\s+|/\s*)?(day|week|month|daily|weekly|monthly)?",
            q,
            flags=re.IGNORECASE,
        )
        if demand_normal_match:
            d_val = _parse_number(demand_normal_match.group(1))
            sd_val = _parse_number(demand_normal_match.group(2))
            demand_period = _detect_time_unit(demand_normal_match.group(3) or q)
            inputs["avg_demand"] = d_val
            inputs["demand_std"] = sd_val
            _record_entity(entities, "avg_demand", "average demand", d_val, demand_normal_match.group(0), f"units/{demand_period}")
            _record_entity(entities, "demand_std", "demand standard deviation", sd_val, demand_normal_match.group(0), f"units/{demand_period}")

        if "avg_demand" not in inputs:
            demand_patterns = [
                (rf"(?:average|mean|expected|constant)?\s*(daily|weekly|monthly)\s+demand\s*(?:averages?|is|of|equals?|=|:)?\s*({_NUM_PATTERN})", 1, 2),
                (rf"(?:average|mean|expected|constant)?\s*demand\s*(?:averages?|is|of|at|equals?|=|:)?\s*(?:constant\s+at\s+)?({_NUM_PATTERN})\s*(?:units?\s*)?(?:per\s+|/\s*|a\s+)(day|week|month)", 2, 1),
                (rf"(?:sells?|uses?|experiences?)\s+(?:an\s+average\s+of\s+)?({_NUM_PATTERN})\s*units?\s*(?:per\s+|/\s*|a\s+)(day|week|month)", 2, 1),
            ]
            for pat, unit_grp, num_grp in demand_patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    demand_period = _detect_time_unit(m.group(unit_grp))
                    d_val = _parse_number(m.group(num_grp))
                    inputs["avg_demand"] = d_val
                    _record_entity(entities, "avg_demand", "average demand", d_val, m.group(0).strip(), f"units/{demand_period}")
                    break

        if "avg_demand" not in inputs:
            # Clause-level fallback for demand with "mean of X" or "averages X"
            for clause in clauses:
                if re.search(r"\bdemand\b", clause, flags=re.IGNORECASE) and not re.search(r"\blead[- ]time\b", clause, flags=re.IGNORECASE):
                    m = re.search(
                        rf"(?:average|mean|expected|averages|constant\s+at)\s*(?:of|is|=|:)?\s*({_NUM_PATTERN})",
                        clause,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        demand_period = _detect_time_unit(clause, default=_detect_time_unit(q))
                        d_val = _parse_number(m.group(1))
                        inputs["avg_demand"] = d_val
                        _record_entity(entities, "avg_demand", "average demand", d_val, m.group(0).strip(), f"units/{demand_period}")
                        break

        if "avg_demand" not in inputs:
            generic_d = _find_first_match(
                q,
                [
                    rf"(?:average|mean|expected|constant)\s+demand\s*(?:is|of|=|:)?\s*({_NUM_PATTERN})",
                    rf"\bd(?:_bar)?\s*=\s*({_NUM_PATTERN})",
                ],
            )
            if generic_d is not None:
                d_val, matched = generic_d
                demand_period = _detect_time_unit(q)
                inputs["avg_demand"] = d_val
                _record_entity(entities, "avg_demand", "average demand", d_val, matched, f"units/{demand_period}")

        inputs["demand_period"] = demand_period

        # 2. Extract average lead time and lead-time standard deviation (explicit + clause-bound)
        lt_normal_match = re.search(
            rf"lead[- ]time\s*(?:is|~|follows)?\s*N\(\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})\s*\)\s*(days?|weeks?|months?)?",
            q,
            flags=re.IGNORECASE,
        )
        if lt_normal_match:
            lt_val = _parse_number(lt_normal_match.group(1))
            lt_sd_val = _parse_number(lt_normal_match.group(2))
            lt_unit = _detect_time_unit(lt_normal_match.group(3) or demand_period, default=demand_period)
            lt_conv, note1 = _convert_time_value(lt_val, lt_unit, demand_period)
            lt_sd_conv, note2 = _convert_time_value(lt_sd_val, lt_unit, demand_period)
            for n in (note1, note2):
                if n:
                    assumptions.append(n)
            inputs["lead_time"] = lt_conv
            inputs["lead_time_std"] = lt_sd_conv
            _record_entity(entities, "lead_time", "average lead time", lt_conv, lt_normal_match.group(0), f"{demand_period}s")
            _record_entity(entities, "lead_time_std", "lead-time standard deviation", lt_sd_conv, lt_normal_match.group(0), f"{demand_period}s")

        # Explicit lead-time std dev patterns
        if "lead_time_std" not in inputs:
            lt_std_patterns = [
                rf"lead[- ]time\s+(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
                rf"(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s+of\s+(?:the\s+)?lead[- ]time\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
                rf"\bsigma_L\s*=\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
            ]
            for pat in lt_std_patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    raw_val = _parse_number(m.group(1))
                    unit = _detect_time_unit(m.group(2) or demand_period, default=demand_period)
                    conv, note = _convert_time_value(raw_val, unit, demand_period)
                    if note:
                        assumptions.append(note)
                    inputs["lead_time_std"] = conv
                    _record_entity(entities, "lead_time_std", "lead-time standard deviation", conv, m.group(0).strip(), f"{demand_period}s")
                    break

        # Explicit average lead time patterns
        if "lead_time" not in inputs:
            lt_patterns = [
                rf"(?:average|mean|expected|constant|replenishment|supplier|delivery)?\s*lead[- ]time\s*(?:averages?|is|of|at|equals?|=|:)?\s*(?:normally\s+distributed\s+with\s+(?:a\s+)?(?:mean|average)\s+(?:of\s+)?)?({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
                rf"(?:delivery|replenishment)\s+time\s*(?:averages?|is|of|=|:)?\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)",
                rf"(?:takes|arrives\s+in)\s+(?:an\s+average\s+of\s+)?({_NUM_PATTERN})\s*(days?|weeks?|months?)",
                rf"\bL\s*=\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
            ]
            for pat in lt_patterns:
                m = re.search(pat, q, flags=re.IGNORECASE)
                if m:
                    raw_val = _parse_number(m.group(1))
                    unit = _detect_time_unit(m.group(2) or demand_period, default=demand_period)
                    conv, note = _convert_time_value(raw_val, unit, demand_period)
                    if note:
                        assumptions.append(note)
                    inputs["lead_time"] = conv
                    _record_entity(entities, "lead_time", "average lead time", conv, m.group(0).strip(), f"{demand_period}s")
                    break

        # 3. Clause-level contextual disambiguation of generic "standard deviation of X"
        # Distinguish whether a generic std-dev phrase belongs to DEMAND vs LEAD TIME!
        for clause in clauses:
            has_demand = bool(re.search(r"\bdemand\b", clause, flags=re.IGNORECASE))
            has_lt = bool(re.search(r"\blead[- ]time\b|\bdelivery\s+time\b", clause, flags=re.IGNORECASE))

            # Handle single sentence with both demand and lead time by splitting around "lead time"
            sub_segments: list[tuple[str, str]] = []
            if has_demand and has_lt:
                parts = re.split(r"(?=\b(?:average\s+|mean\s+|constant\s+)?lead[- ]time\b)", clause, flags=re.IGNORECASE)
                for p in parts:
                    if re.search(r"\blead[- ]time\b", p, flags=re.IGNORECASE):
                        sub_segments.append(("lead_time", p))
                    elif re.search(r"\bdemand\b", p, flags=re.IGNORECASE):
                        sub_segments.append(("demand", p))
            elif has_lt:
                sub_segments.append(("lead_time", clause))
            elif has_demand:
                sub_segments.append(("demand", clause))

            for subject, seg in sub_segments:
                # Remove already-explicitly-labeled std-dev phrases of the opposite subject
                if subject == "lead_time" and "lead_time_std" not in inputs:
                    m = re.search(
                        rf"(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
                        seg,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        raw_val = _parse_number(m.group(1))
                        unit = _detect_time_unit(m.group(2) or demand_period, default=demand_period)
                        conv, note = _convert_time_value(raw_val, unit, demand_period)
                        if note:
                            assumptions.append(note)
                        inputs["lead_time_std"] = conv
                        _record_entity(entities, "lead_time_std", "lead-time standard deviation", conv, m.group(0).strip(), f"{demand_period}s")

                elif subject == "demand" and "demand_std" not in inputs:
                    m = re.search(
                        rf"(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s*(?:of\s+demand\s*)?(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                        seg,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        sd_val = _parse_number(m.group(1))
                        inputs["demand_std"] = sd_val
                        _record_entity(entities, "demand_std", "demand standard deviation", sd_val, m.group(0).strip(), f"units/{demand_period}")

        # Explicit global demand_std fallback (if not captured via clause scan)
        if "demand_std" not in inputs:
            d_std_match = _find_first_match(
                q,
                [
                    rf"(?:daily|weekly|monthly)?\s*demand\s+(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                    rf"(?:standard\s+deviation|std\.?\s*dev\.?|std|sd)\s+of\s+(?:daily|weekly|monthly)?\s*demand\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                    rf"\bsigma_d\s*=\s*({_NUM_PATTERN})",
                ],
            )
            if d_std_match is not None:
                sd_val, matched = d_std_match
                inputs["demand_std"] = sd_val
                _record_entity(entities, "demand_std", "demand standard deviation", sd_val, matched, f"units/{demand_period}")

        # 4. Extract service level, stockout probability, or z-score
        sl_pct = _find_first_match(
            q,
            [
                rf"(?:target\s+|desired\s+)?(?:cycle\s+)?service\s+level\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*%",
                rf"({_NUM_PATTERN})\s*%\s*(?:cycle\s+)?service\s+level",
                rf"(?:in[- ]stock|fill|availability)\s+(?:probability|rate|target|level)\s*(?:is|of|=|:)?\s*({_NUM_PATTERN})\s*%",
                rf"({_NUM_PATTERN})\s*%\s*(?:in[- ]stock|availability)\s*(?:probability|target|level)?",
            ],
        )
        sl_dec = _find_first_match(
            q,
            [
                rf"(?:target\s+|desired\s+)?(?:cycle\s+)?service\s+level\s*(?:is|of|=|:)?\s*(0\.\d+)",
                rf"\bcsl\s*=\s*(0\.\d+)",
            ],
        )
        stockout_pct = _find_first_match(
            q,
            [
                rf"(?:stockout|shortage)\s+(?:probability|risk|rate)\s*(?:is|of|at\s+most|<=|=|:)?\s*({_NUM_PATTERN})\s*%",
                rf"({_NUM_PATTERN})\s*%\s*(?:stockout|shortage)\s+(?:probability|risk|rate)",
            ],
        )
        z_match = _find_first_match(
            q,
            [
                rf"\bz[- ]?(?:score|value|factor)?\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
            ],
        )

        if sl_pct is not None:
            sl_val = sl_pct[0] / 100.0
            inputs["service_level"] = sl_val
            _record_entity(entities, "service_level", "cycle service level", sl_val, sl_pct[1], "probability")
        elif sl_dec is not None:
            inputs["service_level"] = sl_dec[0]
            _record_entity(entities, "service_level", "cycle service level", sl_dec[0], sl_dec[1], "probability")
        elif stockout_pct is not None:
            sl_val = 1.0 - (stockout_pct[0] / 100.0)
            inputs["service_level"] = sl_val
            _record_entity(entities, "service_level", "service level (1 - stockout risk)", sl_val, stockout_pct[1], "probability")
            assumptions.append(
                f"Converted {stockout_pct[0]:g}% stockout risk to {sl_val:.2%} cycle service level."
            )
        elif z_match is not None:
            inputs["z_score"] = z_match[0]
            _record_entity(entities, "z_score", "standard normal z-score", z_match[0], z_match[1], "z")

        return inputs, assumptions, demand_period, entities

    # --- 3. Safety Stock & Reorder Point Semantic Extractor ---

    def _extract_rop_inputs(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        inputs, assumptions, _, entities = self._extract_demand_leadtime_service(q)
        missing: list[str] = []

        if "avg_demand" not in inputs:
            missing.append("avg_demand (average demand per day/week)")
        if "lead_time" not in inputs:
            missing.append("lead_time (average lead time)")

        has_variability = inputs.get("demand_std", 0.0) > 0 or inputs.get("lead_time_std", 0.0) > 0
        if has_variability and "service_level" not in inputs and "z_score" not in inputs:
            missing.append("service_level (e.g., 95%) or z_score")

        return inputs, assumptions, missing, entities

    # --- 4. Fixed-Period Review Semantic Extractor ---

    def _extract_periodic_review_inputs(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        inputs, assumptions, demand_period, entities = self._extract_demand_leadtime_service(q)

        # Review period
        rp_patterns = [
            rf"(?:fixed\s+)?(?:review\s+(?:period|interval|cycle)|order\s+interval)\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
            rf"(?:review(?:ed|s|ing)?|check(?:ed|s|ing)?|order(?:ed|s|ing)?)(?:\s+(?:inventory|stock))?\s+(?:is\s+done\s+|are\s+placed\s+)?every\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)",
            rf"\b(?:P|T)\s*=\s*({_NUM_PATTERN})\s*(days?|weeks?|months?)?",
        ]
        for pat in rp_patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                rp_val = _parse_number(m.group(1))
                rp_unit = _detect_time_unit(m.group(2) or demand_period, default=demand_period)
                rp_conv, note = _convert_time_value(rp_val, rp_unit, demand_period)
                if note:
                    assumptions.append(note)
                inputs["review_period"] = rp_conv
                _record_entity(entities, "review_period", "review period", rp_conv, m.group(0).strip(), f"{demand_period}s")
                break

        # Explicit safety stock (when provided as a numerical value rather than calculated)
        ss_match = _find_first_match(
            q,
            [
                rf"safety\s+stock\s*(?:is|of|equals?|=|:)\s*({_NUM_PATTERN})\s*(?:units?)?",
                rf"({_NUM_PATTERN})\s*units?\s+(?:of\s+)?safety\s+stock",
                rf"\bSS\s*=\s*({_NUM_PATTERN})",
            ],
        )
        if ss_match is not None:
            ss_val, matched = ss_match
            inputs["safety_stock"] = ss_val
            _record_entity(entities, "safety_stock", "explicit safety stock", ss_val, matched, "units")

        # Distinguish: inventory position vs on-hand inventory, on-order inventory, and backorders
        ip_match = _find_first_match(
            q,
            [
                rf"(?:current\s+)?inventory\s+position\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"\bIP\s*=\s*({_NUM_PATTERN})",
            ],
        )
        oh_match = _find_first_match(
            q,
            [
                rf"(?:current(?:ly)?\s+)?on[- ]hand\s+inventory\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"(?:current(?:ly)?\s+)?inventory\s+on[- ]hand\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"(?:on[- ]hand|in[- ]stock)\s*(?:is|of|=|:)\s*({_NUM_PATTERN})",
                rf"({_NUM_PATTERN})\s*units?\s*(?:currently\s+)?(?:on[- ]hand|in\s+stock)",
                rf"(?:currently\s+have|current\s+(?:inventory|stock)\s+(?:is|of))\s*({_NUM_PATTERN})",
                rf"\bOH\s*=\s*({_NUM_PATTERN})",
            ],
        )
        oo_match = _find_first_match(
            q,
            [
                rf"on[- ]order\s+inventory\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"(?:inventory\s+on[- ]order|scheduled\s+receipts?|open\s+orders?|incoming\s+orders?|in[- ]transit(?:\s+inventory)?)\s*(?:is|are|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"on[- ]order\s*(?:is|are|of|=|:)\s*({_NUM_PATTERN})",
                rf"({_NUM_PATTERN})\s*units?\s*(?:currently\s+)?(?:on[- ]order|in[- ]transit|scheduled)",
                rf"\b(?:SR|OO)\s*=\s*({_NUM_PATTERN})",
            ],
        )
        if oo_match is None and re.search(
            r"\bno\s+(?:on[- ]order|scheduled\s+receipts?|open\s+orders?)\b|\bzero\s+on[- ]order\b|\bnothing\s+on[- ]order\b",
            q,
            flags=re.IGNORECASE,
        ):
            oo_match = (0.0, "no on-order inventory")

        bo_match = _find_first_match(
            q,
            [
                rf"backorders?\s*(?:is|are|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                rf"({_NUM_PATTERN})\s*(?:units?\s+)?(?:on\s+backorder|backordered|backorders?)",
                rf"\bBO\s*=\s*({_NUM_PATTERN})",
            ],
        )
        if bo_match is None and re.search(r"\bno\s+backorders?\b|\bzero\s+backorders?\b", q, flags=re.IGNORECASE):
            bo_match = (0.0, "no backorders")

        if ip_match is not None:
            inputs["inventory_position"] = ip_match[0]
            _record_entity(entities, "inventory_position", "inventory position", ip_match[0], ip_match[1], "units")
        else:
            if oh_match is not None:
                inputs["inventory_on_hand"] = oh_match[0]
                _record_entity(entities, "inventory_on_hand", "on-hand inventory", oh_match[0], oh_match[1], "units")
            if oo_match is not None:
                _record_entity(entities, "on_order", "on-order inventory / scheduled receipts", oo_match[0], oo_match[1], "units")
                if oh_match is not None:
                    inputs["on_order"] = oo_match[0]
            if bo_match is not None:
                _record_entity(entities, "backorders", "backorders", bo_match[0], bo_match[1], "units")
                if oh_match is not None:
                    inputs["backorders"] = bo_match[0]

        missing: list[str] = []
        if "avg_demand" not in inputs:
            missing.append("avg_demand (demand per day/week)")
        if "review_period" not in inputs:
            missing.append("review_period (time between reviews)")
        if "lead_time" not in inputs:
            missing.append("lead_time (replenishment lead time)")

        has_variability = inputs.get("demand_std", 0.0) > 0 or inputs.get("lead_time_std", 0.0) > 0
        if has_variability and "safety_stock" not in inputs and "service_level" not in inputs and "z_score" not in inputs:
            missing.append("service_level (e.g., 95%), z_score, or explicit safety_stock")

        return inputs, assumptions, missing, entities

    # --- 5. Newsvendor Semantic Extractor ---

    def _extract_newsvendor_inputs(
        self, q: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
        inputs: dict[str, Any] = {}
        assumptions: list[str] = []
        entities: list[dict[str, Any]] = []

        # Underage / shortage cost
        cu_match = _find_first_match(
            q,
            [
                rf"(?:underage|shortage|stockout|understocking|lost[- ]sales?)\s+cost\s*(?:is|of|equals?|=|:)?\s*\$?\s*({_NUM_PATTERN})",
                rf"cost\s+of\s+(?:underage|a\s+shortage|shortage|understocking)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?(?:underage|shortage|stockout)\s+cost",
                rf"\bC_?u\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )
        # Overage / excess cost
        co_match = _find_first_match(
            q,
            [
                rf"(?:overage|excess|overstocking)\s+cost\s*(?:is|of|equals?|=|:)?\s*\$?\s*({_NUM_PATTERN})",
                rf"cost\s+of\s+(?:overage|excess|overstocking)\s*(?:is|of|=|:)?\s*\$?\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?(?:overage|excess|overstocking)\s+cost",
                rf"\bC_?o\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )

        # Selling price
        sp_match = _find_first_match(
            q,
            [
                rf"(?:unit\s+)?(?:selling|retail|sale|sales)\s+price\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?(?:selling|retail|sale)\s+price",
                rf"(?:sells?|sold|priced)\s+(?:each\s+)?(?:[a-z]+\s+){{0,2}}(?:for|at)\s*\$\s*({_NUM_PATTERN})",
                rf"revenue\s+(?:is|of)\s*\$\s*({_NUM_PATTERN})\s*(?:per\s+unit|each)",
                rf"\bp\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )
        # Purchase cost / unit cost
        uc_match = _find_first_match(
            q,
            [
                rf"(?:unit\s+)?(?:purchase|wholesale|procurement|production|manufacturing|acquisition)\s+(?:cost|price)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"(?:unit\s+cost|cost\s+per\s+unit)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?(?:purchase|unit|wholesale|production)\s+cost",
                rf"(?:costs?|buys?|bought|pays?|paid)\s+(?:each\s+)?(?:[a-z]+\s+){{0,2}}(?:for\s+|at\s+)?\$\s*({_NUM_PATTERN})\s*(?:per\s+unit|each|to\s+(?:buy|make|produce|purchase))?",
                rf"\bc\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )
        # Salvage value (or disposal cost / zero salvage)
        sv_match = _find_first_match(
            q,
            [
                rf"(?:salvage|clearance|markdown|liquidation|residual|scrap|leftover|unsold|buyback)\s+(?:value|price)\s*(?:is|of|equals?|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?(?:salvage|clearance|markdown|scrap|residual)\s+(?:value|price)",
                rf"(?:unsold|leftover|remaining)\s+(?:[a-z]+\s+){{0,3}}(?:are\s+|can\s+be\s+)?(?:sold|salvaged|discounted|marked\s+down|liquidated)\s*(?:for|at)\s*\$\s*({_NUM_PATTERN})",
                rf"salvage(?:d)?\s+(?:for|at)\s*\$\s*({_NUM_PATTERN})",
                rf"\bv\s*=\s*\$?\s*({_NUM_PATTERN})",
            ],
        )
        disposal_match = _find_first_match(
            q,
            [
                rf"disposal\s+(?:cost|fee)\s*(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
                rf"costs?\s*\$\s*({_NUM_PATTERN})\s*(?:per\s+unit\s*)?to\s+(?:dispose|discard)",
            ],
        )
        goodwill_match = _find_first_match(
            q,
            [
                rf"(?:goodwill|penalty)\s+(?:cost\s*)?(?:is|of|=|:)?\s*\$\s*({_NUM_PATTERN})",
            ],
        )

        if cu_match is not None:
            _record_entity(entities, "underage_cost", "underage / shortage cost", cu_match[0], cu_match[1], "$/unit")
        if co_match is not None:
            _record_entity(entities, "overage_cost", "overage / excess cost", co_match[0], co_match[1], "$/unit")
        if sp_match is not None:
            _record_entity(entities, "selling_price", "selling price", sp_match[0], sp_match[1], "$/unit")
        if uc_match is not None:
            _record_entity(entities, "unit_cost", "purchase / unit cost", uc_match[0], uc_match[1], "$/unit")
        if sv_match is not None:
            _record_entity(entities, "salvage_value", "salvage value", sv_match[0], sv_match[1], "$/unit")
        elif disposal_match is not None:
            sv_match = (-disposal_match[0], disposal_match[1])
            _record_entity(entities, "salvage_value", "negative salvage value (disposal cost)", -disposal_match[0], disposal_match[1], "$/unit")
            assumptions.append(f"Represented disposal cost of ${disposal_match[0]:,.2f}/unit as salvage_value = -${disposal_match[0]:,.2f}.")
        elif re.search(
            r"\bno\s+salvage\b|\bzero\s+salvage\b|\bsalvage\s+value\s+(?:is\s+|of\s+)?\$?0\b|\bthrown\s+away\b|\bdiscarded\b|\bworthless\b|\bdonated\s+for\s+nothing\b",
            q,
            flags=re.IGNORECASE,
        ):
            sv_match = (0.0, "no salvage value / discarded")
            _record_entity(entities, "salvage_value", "salvage value (zero)", 0.0, "no salvage value", "$/unit")

        if cu_match is not None and co_match is not None:
            inputs["underage_cost"] = cu_match[0]
            inputs["overage_cost"] = co_match[0]
        elif sp_match is not None and uc_match is not None:
            inputs["selling_price"] = sp_match[0]
            inputs["unit_cost"] = uc_match[0]
            if sv_match is not None:
                inputs["salvage_value"] = sv_match[0]
            if goodwill_match is not None:
                inputs["goodwill_penalty"] = goodwill_match[0]
                _record_entity(entities, "goodwill_penalty", "goodwill penalty", goodwill_match[0], goodwill_match[1], "$/unit")

        # Demand distribution detection: Discrete vs Uniform vs Normal
        discrete_scenarios = _extract_discrete_scenarios(q)
        uniform_match = re.search(
            rf"(?:uniform(?:ly)?\s*(?:distributed\s*)?(?:between|from|on)?|U)\s*[\[\(]?\s*({_NUM_PATTERN})\s*(?:and|to|-|,)\s*({_NUM_PATTERN})\s*[\]\)]?",
            q,
            flags=re.IGNORECASE,
        )

        if len(discrete_scenarios) >= 2:
            inputs["distribution"] = "discrete"
            inputs["demand_scenarios"] = discrete_scenarios
            _record_entity(entities, "distribution", "demand distribution", "discrete", "discrete scenarios")
            _record_entity(entities, "demand_scenarios", "discrete demand scenarios", discrete_scenarios, f"{len(discrete_scenarios)} scenarios")
        elif uniform_match:
            min_d = _parse_number(uniform_match.group(1))
            max_d = _parse_number(uniform_match.group(2))
            inputs["distribution"] = "uniform"
            inputs["min_demand"] = min_d
            inputs["max_demand"] = max_d
            _record_entity(entities, "distribution", "demand distribution", "uniform", uniform_match.group(0).strip())
            _record_entity(entities, "min_demand", "uniform minimum demand", min_d, uniform_match.group(0).strip(), "units")
            _record_entity(entities, "max_demand", "uniform maximum demand", max_d, uniform_match.group(0).strip(), "units")
        else:
            inputs["distribution"] = "normal"
            _record_entity(entities, "distribution", "demand distribution", "normal", "normal distribution")

            normal_compact = re.search(r"N\(\s*(" + _NUM_PATTERN + r")\s*,\s*(" + _NUM_PATTERN + r")\s*\)", q)
            if normal_compact:
                mu_val = _parse_number(normal_compact.group(1))
                sd_val = _parse_number(normal_compact.group(2))
                inputs["mean_demand"] = mu_val
                inputs["demand_std"] = sd_val
                _record_entity(entities, "mean_demand", "mean demand", mu_val, normal_compact.group(0), "units")
                _record_entity(entities, "demand_std", "demand standard deviation", sd_val, normal_compact.group(0), "units")
            else:
                mean_match = _find_first_match(
                    q,
                    [
                        rf"(?:mean|average|expected)\s+(?:season(?:al)?\s+|daily\s+|weekly\s+)?demand\s*(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                        rf"demand\s+(?:averages?|is\s+(?:normally\s+distributed\s+)?(?:with\s+)?(?:a\s+)?(?:mean|average)\s*(?:of|is|=|:)?)\s*({_NUM_PATTERN})",
                        rf"(?:mean|average|expected)\s*(?:is|of|=|:)?\s*({_NUM_PATTERN})",
                        rf"\b(?:mu|μ)\s*=\s*({_NUM_PATTERN})",
                    ],
                )
                std_match = _find_first_match(
                    q,
                    [
                        rf"(?:demand\s+)?(?:standard\s+deviation|std\.?\s*dev\.?|std|sd|sigma)\s*(?:of\s+demand\s*)?(?:is|of|equals?|=|:)?\s*({_NUM_PATTERN})",
                    ],
                )
                if mean_match is not None:
                    inputs["mean_demand"] = mean_match[0]
                    _record_entity(entities, "mean_demand", "mean demand", mean_match[0], mean_match[1], "units")
                if std_match is not None:
                    inputs["demand_std"] = std_match[0]
                    _record_entity(entities, "demand_std", "demand standard deviation", std_match[0], std_match[1], "units")

        # Missing check only AFTER full-question extraction
        missing: list[str] = []
        has_direct_costs = "underage_cost" in inputs and "overage_cost" in inputs
        has_econ_costs = "selling_price" in inputs and "unit_cost" in inputs and "salvage_value" in inputs
        if not (has_direct_costs or has_econ_costs):
            if "selling_price" in inputs and "unit_cost" in inputs and "salvage_value" not in inputs:
                missing.append("salvage_value ($ per unsold unit, or state $0 / no salvage value)")
            else:
                missing.append("selling_price, purchase cost, and salvage_value (or underage_cost and overage_cost)")

        if inputs["distribution"] == "normal":
            if "mean_demand" not in inputs:
                missing.append("mean_demand")
            if "demand_std" not in inputs:
                missing.append("demand_std")

        return inputs, assumptions, missing, entities

    def _extract_inputs_with_llm(
        self, question: str, model: str
    ) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]] | None:
        """Optional OpenAI JSON extractor (never performs calculations)."""
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            return None

        client = OpenAI()
        schema_hint = TOOL_REGISTRY[model]["required_inputs"]
        prompt = (
            f"Extract numeric parameters and their meanings for the '{model}' supply chain inventory model.\n"
            f"Expected fields: {schema_hint}.\n"
            "Examine the entire question before marking anything missing. NEVER invent missing numbers.\n"
            "Return ONLY valid JSON with keys: 'inputs' (dict), 'assumptions' (list of str), "
            "'missing_inputs' (list of str), 'entities' (list of {parameter, meaning, value, matched_text}).\n"
            f"Question: {question}"
        )
        try:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            return (
                data.get("inputs", {}),
                data.get("assumptions", []),
                data.get("missing_inputs", []),
                data.get("entities", []),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Stage 4: Select Python Tool
    # ------------------------------------------------------------------

    def select_tool(self, model: str) -> tuple[str, Any]:
        """Select the registered Python tool function for the identified model."""
        if model not in TOOL_REGISTRY:
            raise ValueError(f"No Python tool registered for model '{model}'.")
        entry = TOOL_REGISTRY[model]
        return entry["tool_name"], entry["function"]

    # ------------------------------------------------------------------
    # Stage 5: Calculate
    # ------------------------------------------------------------------

    def calculate(self, model: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the selected Python tool with extracted inputs."""
        _, func = self.select_tool(model)
        return func(**inputs)

    # ------------------------------------------------------------------
    # Stage 6: Verify
    # ------------------------------------------------------------------

    def verify(self, model: str, result: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic verification checks produced by the Python tool."""
        if "verification_checks" in result:
            return result["verification_checks"]
        return verify_calculation(model, result)

    # ------------------------------------------------------------------
    # Stage 7: Explain (Renders the 8 Deterministic Python Tool Sections Verbatim)
    # ------------------------------------------------------------------

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:,.4f}".rstrip("0").rstrip(".")
        if isinstance(value, list):
            if value and isinstance(value[0], dict) and "unit_price" in value[0]:
                parts = []
                for idx, t in enumerate(value, start=1):
                    rng = (
                        f"{t['min_qty']:g}-{t['max_qty']:g}"
                        if t.get("max_qty") is not None
                        else f"{t['min_qty']:g}+"
                    )
                    parts.append(f"Tier {idx} ({rng} @ ${t['unit_price']:,.2f})")
                return "; ".join(parts)
            return str(value)
        return str(value)

    def explain(
        self,
        model: str,
        result: dict[str, Any],
        verification: dict[str, Any],
        assumptions: list[str],
    ) -> str:
        """Format the 8 student-verifiable sections directly from the Python tool output.

        Never performs or reconstructs any numerical calculations outside inventory_tools.py.
        """
        display_name = TOOL_REGISTRY[model]["display_name"]
        badge = (
            f"Verified ({verification['checks_passed']}/{verification['total_checks']} checks passed)"
            if verification["verified"]
            else f"Verification Warning ({verification['checks_passed']}/{verification['total_checks']} checks passed)"
        )

        lines: list[str] = [f"### {display_name} — {badge}", ""]

        # 1. Inputs Extracted
        lines.append("#### 1. Inputs Extracted")
        for k, v in result["inputs"].items():
            lines.append(f"- `{k}`: {self._format_value(v)}")
        lines.append("")

        # 2. Variable Definitions
        lines.append("#### 2. Variable Definitions")
        for sym, desc in result["variable_definitions"].items():
            lines.append(f"- `{sym}`: {desc}")
        lines.append("")

        # 3. Formula
        lines.append("#### 3. Formula")
        for f_str in result["formulas"]:
            lines.append(f"- `{f_str}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("#### 4. Formula with Substituted Values")
        for s_str in result["substituted_formulas"]:
            lines.append(f"- {s_str}")
        lines.append("")

        # 5. Intermediate Calculations
        lines.append("#### 5. Intermediate Calculations")
        for k, v in result["intermediate_calculations"].items():
            if k == "tier_evaluations" and isinstance(v, list):
                lines.append("- `tier_evaluations`:")
                for t in v:
                    if t["max_qty"] is not None:
                        rng = f"{t['min_qty']:,.0f}–{t['max_qty']:,.0f}"
                    elif t.get("upper_bound_exclusive") is not None:
                        rng = f"{t['min_qty']:,.0f} to <{t['upper_bound_exclusive']:,.0f}"
                    else:
                        rng = f"{t['min_qty']:,.0f}+"
                    if t["is_feasible"]:
                        lines.append(
                            f"  - Tier {t['tier']} ({rng} @ ${t['unit_price']:,.2f}, H=${t['holding_cost']:,.4f}): "
                            f"raw_eoq={t['raw_eoq']:,.2f}, evaluated_qty={t['evaluated_order_qty']:,.2f} "
                            f"({t['status']}), total_annual_cost=${t['total_annual_cost']:,.2f}"
                        )
                    else:
                        lines.append(
                            f"  - Tier {t['tier']} ({rng} @ ${t['unit_price']:,.2f}, H=${t['holding_cost']:,.4f}): "
                            f"raw_eoq={t['raw_eoq']:,.2f} ({t['status']})"
                        )
            else:
                lines.append(f"- `{k}`: {self._format_value(v)}")
        lines.append("")

        # 6. Final Answer
        lines.append("#### 6. Final Answer")
        for k, v in result["final_answer"].items():
            lines.append(f"- **{k}**: {self._format_value(v)}")
        lines.append("")

        # 7. Verification Checks
        lines.append("#### 7. Verification Checks")
        for chk in verification["checks"]:
            mark = "PASS" if chk["passed"] else "FAIL"
            lines.append(f"- [{mark}] `{chk['check']}`: {chk['detail']}")
        lines.append("")

        # 8. Assumptions
        lines.append("#### 8. Assumptions")
        for a in assumptions:
            lines.append(f"- {a}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full Pipeline Orchestrator
    # ------------------------------------------------------------------

    def answer(self, question: str) -> AgentResponse:
        """Execute the full 7-stage pipeline on a natural-language inventory question."""
        parser_mode = "llm_assisted" if self.use_llm else "deterministic"

        # Stage 2: Identify Model
        model, confidence, signals = self.identify_model(question)
        if model is None:
            return AgentResponse(
                question=question,
                identified_model=None,
                model_display_name=None,
                identification_confidence=0.0,
                identification_signals=signals,
                status="needs_clarification",
                parser_mode=parser_mode,
                explanation=(
                    "I could not confidently identify which inventory model applies to your question. "
                    "Please clarify whether this is an **EOQ**, **Quantity Discount**, **Safety Stock / Reorder Point**, "
                    "**Fixed-Period Review**, or **Newsvendor** problem, and include the relevant demand and cost parameters."
                ),
            )

        display_name = TOOL_REGISTRY[model]["display_name"]

        # Stage 3: Extract Inputs (full-question entity + meaning scan before checking missing inputs)
        extracted, parser_assumptions, missing, entities = self.extract_inputs(question, model)
        tool_name, _ = self.select_tool(model)

        if missing:
            missing_bullets = "\n".join(f"- {m}" for m in missing)
            return AgentResponse(
                question=question,
                identified_model=model,
                model_display_name=display_name,
                identification_confidence=confidence,
                identification_signals=signals,
                extracted_entities=entities,
                extracted_inputs=extracted,
                assumptions=parser_assumptions,
                missing_inputs=missing,
                selected_tool=tool_name,
                status="needs_clarification",
                parser_mode=parser_mode,
                explanation=(
                    f"Identified **{display_name}**, but I need the following missing input(s) before running `{tool_name}`:\n"
                    f"{missing_bullets}"
                ),
            )

        # Stage 5: Calculate
        try:
            calc_result = self.calculate(model, extracted)
        except Exception as exc:
            return AgentResponse(
                question=question,
                identified_model=model,
                model_display_name=display_name,
                identification_confidence=confidence,
                identification_signals=signals,
                extracted_entities=entities,
                extracted_inputs=extracted,
                assumptions=parser_assumptions,
                selected_tool=tool_name,
                status="error",
                error_message=str(exc),
                parser_mode=parser_mode,
                explanation=f"Calculation failed for **{display_name}** (`{tool_name}`): {exc}",
            )

        all_assumptions = parser_assumptions + calc_result.get("assumptions", [])

        # Stage 6: Verify
        verification = self.verify(model, calc_result)

        # Stage 7: Explain
        explanation = self.explain(model, calc_result, verification, all_assumptions)

        return AgentResponse(
            question=question,
            identified_model=model,
            model_display_name=display_name,
            identification_confidence=confidence,
            identification_signals=signals,
            extracted_entities=entities,
            extracted_inputs=extracted,
            assumptions=all_assumptions,
            missing_inputs=[],
            selected_tool=tool_name,
            calculation_result=calc_result,
            verification=verification,
            explanation=explanation,
            status="ok",
            parser_mode=parser_mode,
        )
