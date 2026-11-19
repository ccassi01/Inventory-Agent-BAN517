"""Deterministic Python calculation tools and verification routines for supply chain inventory models.

All numerical calculations, substituted formulas, intermediate steps, final answers,
and verification checks are produced exclusively and deterministically in Python.

Tools:
    1. calculate_eoq()
    2. calculate_quantity_discount()
    3. calculate_rop()
    4. calculate_periodic_review()
    5. calculate_newsvendor()

Every tool returns all 8 student-verifiable sections:
    1. inputs                     (Inputs Extracted)
    2. variable_definitions       (Variable Definitions)
    3. formulas                   (Formula)
    4. substituted_formulas       (Formula with Substituted Values)
    5. intermediate_calculations  (Intermediate Calculations)
    6. final_answer               (Final Answer)
    7. verification_checks        (Verification Checks)
    8. assumptions                (Assumptions)
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

_STD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


# ---------------------------------------------------------------------------
# Strict Input Validation Helpers
# ---------------------------------------------------------------------------


def _ensure_finite_number(name: str, value: Any) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a finite numeric value (got {value!r}).")
    val = float(value)
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"'{name}' must be a finite numeric value (got {value!r}).")
    return val


def _validate_positive(name: str, value: Any) -> float:
    val = _ensure_finite_number(name, value)
    if val <= 0.0:
        raise ValueError(f"'{name}' must be strictly positive (got {val}).")
    return val


def _validate_non_negative(name: str, value: Any) -> float:
    val = _ensure_finite_number(name, value)
    if val < 0.0:
        raise ValueError(f"'{name}' must be non-negative (got {val}).")
    return val


def _normalize_rate(name: str, value: Any) -> tuple[float, bool]:
    """Validate a rate/percentage. Accepts fraction (0 < r <= 1) or percentage (1 < r < 100)."""
    val = _validate_positive(name, value)
    if val > 1.0:
        if val >= 100.0:
            raise ValueError(f"'{name}' percentage must be less than 100 (got {val}).")
        return val / 100.0, True
    return val, False


def _resolve_service_level_and_z(
    service_level: float | None = None,
    z_score: float | None = None,
) -> tuple[float, float, list[str]]:
    """Resolve (service_level, z_score, notes) strictly from provided input(s)."""
    notes: list[str] = []
    if service_level is None and z_score is None:
        raise ValueError(
            "Missing required service target: provide either 'service_level' or 'z_score'."
        )

    if service_level is not None and z_score is not None:
        sl_raw = _ensure_finite_number("service_level", service_level)
        sl = sl_raw / 100.0 if 1.0 < sl_raw < 100.0 else sl_raw
        if not (0.0 < sl < 1.0):
            raise ValueError(
                f"'service_level' must be strictly between 0 and 1 (or 1 and 100 as %), got {service_level}."
            )
        z = _ensure_finite_number("z_score", z_score)
        expected_z = _STD_NORMAL.inv_cdf(sl)
        if not math.isclose(z, expected_z, rel_tol=1e-2, abs_tol=0.02):
            raise ValueError(
                f"Provided 'service_level' ({sl:.4f} => z={expected_z:.4f}) conflicts with 'z_score' ({z:.4f})."
            )
        return sl, z, notes

    if service_level is not None:
        sl_raw = _ensure_finite_number("service_level", service_level)
        if 1.0 < sl_raw < 100.0:
            sl = sl_raw / 100.0
            notes.append(f"Interpreted service_level={sl_raw:g} as {sl:.4%} ({sl:.6g}).")
        else:
            sl = sl_raw
        if not (0.0 < sl < 1.0):
            raise ValueError(
                f"'service_level' must be strictly between 0 and 1 (or 1 and 100 as %), got {service_level}."
            )
        z = _STD_NORMAL.inv_cdf(sl)
        return sl, z, notes

    assert z_score is not None
    z = _ensure_finite_number("z_score", z_score)
    sl = _STD_NORMAL.cdf(z)
    return sl, z, notes


def _fmt(val: float, decimals: int = 4) -> str:
    """Compact human-readable numeric formatting without losing verification precision."""
    if math.isclose(val, round(val), abs_tol=1e-9):
        return f"{round(val):,}"
    return f"{val:,.{decimals}f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# ---------------------------------------------------------------------------


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost: float | None = None,
    unit_cost: float | None = None,
    holding_cost_rate: float | None = None,
    working_days_per_year: float | None = None,
) -> dict[str, Any]:
    """Compute the Economic Order Quantity (EOQ): Q* = sqrt(2DS / H)."""
    D = _validate_positive("annual_demand", annual_demand)
    S = _validate_positive("ordering_cost", ordering_cost)

    assumptions: list[str] = [
        "Demand is constant, known, and uniform throughout the year.",
        "Ordering cost per order and annual per-unit holding cost are constant.",
        "Replenishment arrives instantaneously as a single batch with no stockouts or quantity discounts.",
    ]

    C: float | None = None
    if unit_cost is not None:
        C = _validate_positive("unit_cost", unit_cost)

    rate: float | None = None
    if holding_cost is not None:
        H = _validate_positive("holding_cost", holding_cost)
        holding_cost_source = "direct"
    elif C is not None and holding_cost_rate is not None:
        rate, was_pct = _normalize_rate("holding_cost_rate", holding_cost_rate)
        H = C * rate
        holding_cost_source = "derived_from_unit_cost_and_rate"
        if was_pct:
            assumptions.append(f"Interpreted holding_cost_rate as {rate:.2%} of unit cost (${C:,.2f}).")
    else:
        raise ValueError(
            "Missing required holding cost: provide either 'holding_cost' "
            "or both 'unit_cost' and 'holding_cost_rate'."
        )

    days_per_year: float | None = None
    if working_days_per_year is not None:
        days_per_year = _validate_positive("working_days_per_year", working_days_per_year)

    numerator_2DS = 2.0 * D * S
    eoq = math.sqrt(numerator_2DS / H)
    orders_per_year = D / eoq
    cycle_time_years = eoq / D
    average_inventory = eoq / 2.0
    annual_ordering_cost = orders_per_year * S
    annual_holding_cost = average_inventory * H
    total_inventory_cost = annual_ordering_cost + annual_holding_cost

    inputs: dict[str, Any] = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost": H,
    }
    if C is not None:
        inputs["unit_cost"] = C
    if rate is not None:
        inputs["holding_cost_rate"] = rate
    if days_per_year is not None:
        inputs["working_days_per_year"] = days_per_year

    variable_definitions: dict[str, str] = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / setup cost per order ($/order)",
        "H": "Annual holding cost per unit ($/unit/year)",
        "Q*": "Economic Order Quantity (units/order)",
        "N": "Expected number of orders per year (orders/year)",
        "T": "Time between orders / cycle length (years)",
        "Avg_Inv": "Average cycle inventory = Q* / 2 (units)",
        "TC_inv": "Total annual relevant inventory cost = Ordering Cost + Holding Cost ($/year)",
    }
    if C is not None:
        variable_definitions["C"] = "Unit purchase cost ($/unit)"
        variable_definitions["TC"] = "Total annual cost including purchasing ($/year)"
    if rate is not None:
        variable_definitions["I"] = "Annual holding cost rate (fraction of unit cost)"
    if days_per_year is not None:
        variable_definitions["W"] = "Working days per year (days/year)"

    formulas: list[str] = []
    substituted_formulas: list[str] = []

    if rate is not None and C is not None:
        formulas.append("H = I * C")
        substituted_formulas.append(f"H = {rate:.4f} * ${C:,.2f} = ${H:,.4f}/unit/year")

    formulas.extend(
        [
            "Q* = sqrt((2 * D * S) / H)",
            "N = D / Q*",
            "T = Q* / D",
            "Avg_Inv = Q* / 2",
            "Annual Ordering Cost = (D / Q*) * S",
            "Annual Holding Cost = (Q* / 2) * H",
            "TC_inv = Annual Ordering Cost + Annual Holding Cost",
        ]
    )
    substituted_formulas.extend(
        [
            f"Q* = sqrt((2 * {_fmt(D)} * ${S:,.2f}) / ${H:,.4f}) = sqrt({numerator_2DS:,.2f} / {H:,.4f}) = {eoq:,.2f} units",
            f"N = {_fmt(D)} / {eoq:,.2f} = {orders_per_year:,.2f} orders/year",
            f"T = {eoq:,.2f} / {_fmt(D)} = {cycle_time_years:.4f} years",
            f"Avg_Inv = {eoq:,.2f} / 2 = {average_inventory:,.2f} units",
            f"Annual Ordering Cost = ({_fmt(D)} / {eoq:,.2f}) * ${S:,.2f} = ${annual_ordering_cost:,.2f}/year",
            f"Annual Holding Cost = ({eoq:,.2f} / 2) * ${H:,.4f} = ${annual_holding_cost:,.2f}/year",
            f"TC_inv = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_inventory_cost:,.2f}/year",
        ]
    )

    intermediate: dict[str, Any] = {
        "holding_cost_source": holding_cost_source,
        "holding_cost_H": H,
        "numerator_2DS": numerator_2DS,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "average_inventory": average_inventory,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
    }
    if days_per_year is not None:
        cycle_days = cycle_time_years * days_per_year
        intermediate["cycle_time_days"] = cycle_days
        formulas.append("T_days = (Q* / D) * W")
        substituted_formulas.append(
            f"T_days = ({eoq:,.2f} / {_fmt(D)}) * {_fmt(days_per_year)} = {cycle_days:,.2f} days"
        )

    final_answer: dict[str, Any] = {
        "eoq": eoq,
        "recommended_order_quantity": round(eoq),
        "total_inventory_cost": total_inventory_cost,
    }

    if C is not None:
        annual_purchase_cost = D * C
        total_annual_cost = annual_purchase_cost + total_inventory_cost
        intermediate["annual_purchase_cost"] = annual_purchase_cost
        final_answer["total_annual_cost"] = total_annual_cost
        formulas.append("TC = (D * C) + TC_inv")
        substituted_formulas.append(
            f"TC = ({_fmt(D)} * ${C:,.2f}) + ${total_inventory_cost:,.2f} = ${annual_purchase_cost:,.2f} + ${total_inventory_cost:,.2f} = ${total_annual_cost:,.2f}/year"
        )

    result: dict[str, Any] = {
        "model": "eoq",
        "inputs": inputs,
        "variable_definitions": variable_definitions,
        "formulas": formulas,
        "substituted_formulas": substituted_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
    }
    result["verification_checks"] = verify_calculation("eoq", result)
    return result


# ---------------------------------------------------------------------------
# 2. Quantity Discount (All-Units)
# ---------------------------------------------------------------------------


def calculate_quantity_discount(
    annual_demand: float,
    ordering_cost: float,
    price_breaks: list[dict[str, Any]],
    holding_cost: float | None = None,
    holding_cost_rate: float | None = None,
) -> dict[str, Any]:
    """Evaluate All-Units Quantity Discount price breaks and minimize total annual cost."""
    D = _validate_positive("annual_demand", annual_demand)
    S = _validate_positive("ordering_cost", ordering_cost)

    if not isinstance(price_breaks, list) or len(price_breaks) == 0:
        raise ValueError("'price_breaks' must be a non-empty list of tier dictionaries.")

    if holding_cost is not None and holding_cost_rate is not None:
        raise ValueError("Provide either 'holding_cost' (fixed) OR 'holding_cost_rate' (percentage), not both.")

    fixed_H: float | None = None
    rate: float | None = None
    assumptions: list[str] = [
        "All-units quantity discount model: the unit price of the chosen tier applies to all units in the order.",
        "Annual demand and ordering cost are constant and deterministic.",
    ]

    if holding_cost is not None:
        fixed_H = _validate_positive("holding_cost", holding_cost)
        holding_cost_type = "fixed"
        assumptions.append(f"Holding cost is constant at ${fixed_H:,.4f}/unit/year across all price tiers.")
    elif holding_cost_rate is not None:
        rate, _ = _normalize_rate("holding_cost_rate", holding_cost_rate)
        holding_cost_type = "percentage"
        assumptions.append(
            f"Annual holding cost scales with unit price at {rate:.2%} of each tier's unit price (H_i = I * C_i)."
        )
    else:
        holding_cost_type = "per_tier"

    normalized_tiers: list[dict[str, Any]] = []
    for idx, tier in enumerate(price_breaks):
        if not isinstance(tier, dict) or "min_qty" not in tier or "unit_price" not in tier:
            raise ValueError(f"price_breaks[{idx}] must be a dict with 'min_qty' and 'unit_price'.")
        min_q = _validate_non_negative(f"price_breaks[{idx}].min_qty", tier["min_qty"])
        max_q_raw = tier.get("max_qty")
        max_q = _validate_positive(f"price_breaks[{idx}].max_qty", max_q_raw) if max_q_raw is not None else None
        if max_q is not None and max_q < min_q:
            raise ValueError(f"price_breaks[{idx}] has max_qty ({max_q}) < min_qty ({min_q}).")
        price = _validate_positive(f"price_breaks[{idx}].unit_price", tier["unit_price"])

        if "holding_cost" in tier and tier["holding_cost"] is not None:
            tier_H = _validate_positive(f"price_breaks[{idx}].holding_cost", tier["holding_cost"])
        elif rate is not None:
            tier_H = rate * price
        elif fixed_H is not None:
            tier_H = fixed_H
        else:
            raise ValueError(
                "Missing required holding cost: provide 'holding_cost' (fixed), "
                "'holding_cost_rate' (percentage), or per-tier 'holding_cost'."
            )

        normalized_tiers.append(
            {
                "min_qty": min_q,
                "max_qty": max_q,
                "unit_price": price,
                "holding_cost": tier_H,
            }
        )

    normalized_tiers.sort(key=lambda t: t["min_qty"])

    for i in range(1, len(normalized_tiers)):
        if normalized_tiers[i]["min_qty"] <= normalized_tiers[i - 1]["min_qty"]:
            raise ValueError("Price break tiers must have strictly increasing 'min_qty' thresholds.")

    tier_evaluations: list[dict[str, Any]] = []
    feasible_candidates: list[dict[str, Any]] = []
    substituted_formulas: list[str] = []

    for idx, tier in enumerate(normalized_tiers):
        tier_num = idx + 1
        min_q = tier["min_qty"]
        explicit_max = tier["max_qty"]
        next_min = normalized_tiers[idx + 1]["min_qty"] if idx + 1 < len(normalized_tiers) else None
        price = tier["unit_price"]
        H_i = tier["holding_cost"]

        raw_eoq = math.sqrt((2.0 * D * S) / H_i)

        exceeds_max = False
        if explicit_max is not None and raw_eoq > explicit_max:
            exceeds_max = True
        elif explicit_max is None and next_min is not None and raw_eoq >= next_min:
            exceeds_max = True

        if exceeds_max:
            status = "infeasible_above_max"
            is_feasible = False
            evaluated_qty = None
        elif raw_eoq < min_q:
            status = "adjusted_to_min_qty"
            is_feasible = True
            evaluated_qty = min_q
        else:
            status = "feasible_eoq"
            is_feasible = True
            evaluated_qty = raw_eoq

        if rate is not None:
            substituted_formulas.append(
                f"Tier {tier_num}: H_{tier_num} = {rate:.4f} * ${price:,.2f} = ${H_i:,.4f}/unit/year"
            )
        substituted_formulas.append(
            f"Tier {tier_num}: Q_{tier_num} = sqrt((2 * {_fmt(D)} * ${S:,.2f}) / ${H_i:,.4f}) = {raw_eoq:,.2f} units ({status})"
        )

        entry: dict[str, Any] = {
            "tier": tier_num,
            "min_qty": min_q,
            "max_qty": explicit_max,
            "upper_bound_exclusive": next_min if explicit_max is None else None,
            "unit_price": price,
            "holding_cost": H_i,
            "raw_eoq": raw_eoq,
            "status": status,
            "is_feasible": is_feasible,
            "evaluated_order_qty": evaluated_qty,
            "annual_purchase_cost": None,
            "annual_ordering_cost": None,
            "annual_holding_cost": None,
            "total_annual_cost": None,
            "orders_per_year": None,
        }

        if is_feasible and evaluated_qty is not None:
            purchase_cost = D * price
            ordering_total = (D / evaluated_qty) * S
            holding_total = (evaluated_qty / 2.0) * H_i
            total_cost = purchase_cost + ordering_total + holding_total
            entry.update(
                {
                    "annual_purchase_cost": purchase_cost,
                    "annual_ordering_cost": ordering_total,
                    "annual_holding_cost": holding_total,
                    "total_annual_cost": total_cost,
                    "orders_per_year": D / evaluated_qty,
                }
            )
            feasible_candidates.append(entry)
            substituted_formulas.append(
                f"Tier {tier_num}: TC({evaluated_qty:,.2f}) = ({_fmt(D)} * ${price:,.2f}) + ({_fmt(D)} / {evaluated_qty:,.2f}) * ${S:,.2f} + ({evaluated_qty:,.2f} / 2) * ${H_i:,.4f} = ${purchase_cost:,.2f} + ${ordering_total:,.2f} + ${holding_total:,.2f} = ${total_cost:,.2f}/year"
            )
        else:
            substituted_formulas.append(
                f"Tier {tier_num}: Infeasible (Q_{tier_num} = {raw_eoq:,.2f} exceeds tier upper bound)"
            )

        tier_evaluations.append(entry)

    if not feasible_candidates:
        raise ValueError("No feasible order quantity exists across the supplied price-break tiers.")

    best = min(feasible_candidates, key=lambda c: c["total_annual_cost"])
    first_feasible = feasible_candidates[0]
    savings_vs_base_tier = first_feasible["total_annual_cost"] - best["total_annual_cost"]

    comparison_str = ", ".join(
        f"Tier {c['tier']}: ${c['total_annual_cost']:,.2f}" for c in feasible_candidates
    )
    substituted_formulas.append(
        f"Optimal Selection = min({comparison_str}) -> Tier {best['tier']} (Q* = {best['evaluated_order_qty']:,.2f} units, TC = ${best['total_annual_cost']:,.2f}/year)"
    )

    inputs: dict[str, Any] = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost_type": holding_cost_type,
        "price_breaks": normalized_tiers,
    }
    if fixed_H is not None:
        inputs["holding_cost"] = fixed_H
    if rate is not None:
        inputs["holding_cost_rate"] = rate

    variable_definitions: dict[str, str] = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / setup cost per order ($/order)",
        "C_i": "Unit purchase price in price-break tier i ($/unit)",
        "H_i": "Annual holding cost per unit in tier i ($/unit/year)",
        "Q_i": "Unconstrained EOQ for tier i (units)",
        "Q_adj,i": "Feasible evaluated order quantity for tier i (units)",
        "TC(Q)": "Total annual cost = Purchase Cost + Ordering Cost + Holding Cost ($/year)",
    }
    if rate is not None:
        variable_definitions["I"] = "Annual percentage holding cost rate (fraction of unit price)"
    if fixed_H is not None:
        variable_definitions["H"] = "Fixed annual holding cost per unit ($/unit/year)"

    formulas: list[str] = []
    if rate is not None:
        formulas.append("H_i = I * C_i")
    elif fixed_H is not None:
        formulas.append("H_i = H (constant across all tiers)")
    formulas.extend(
        [
            "Q_i = sqrt((2 * D * S) / H_i)",
            "Feasibility: if Q_i < min_qty_i then Q_adj,i = min_qty_i; if Q_i > max_qty_i then infeasible; else Q_adj,i = Q_i",
            "TC(Q_adj,i) = (D * C_i) + (D / Q_adj,i) * S + (Q_adj,i / 2) * H_i",
            "Optimal Tier = argmin_i TC(Q_adj,i)",
        ]
    )

    intermediate: dict[str, Any] = {
        "holding_cost_type": holding_cost_type,
        "tier_evaluations": tier_evaluations,
        "feasible_tier_count": len(feasible_candidates),
        "optimal_annual_purchase_cost": best["annual_purchase_cost"],
        "optimal_annual_ordering_cost": best["annual_ordering_cost"],
        "optimal_annual_holding_cost": best["annual_holding_cost"],
        "optimal_orders_per_year": best["orders_per_year"],
        "savings_vs_base_tier": savings_vs_base_tier,
    }

    final_answer: dict[str, Any] = {
        "optimal_tier": best["tier"],
        "optimal_order_quantity": best["evaluated_order_qty"],
        "recommended_order_quantity": round(best["evaluated_order_qty"]),
        "optimal_unit_price": best["unit_price"],
        "optimal_holding_cost": best["holding_cost"],
        "total_annual_cost": best["total_annual_cost"],
    }

    result: dict[str, Any] = {
        "model": "quantity_discount",
        "inputs": inputs,
        "variable_definitions": variable_definitions,
        "formulas": formulas,
        "substituted_formulas": substituted_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
    }
    result["verification_checks"] = verify_calculation("quantity_discount", result)
    return result


# ---------------------------------------------------------------------------
# 3. Safety Stock and Reorder Point (Continuous Review)
# ---------------------------------------------------------------------------


def calculate_rop(
    avg_demand: float,
    lead_time: float,
    demand_std: float | None = None,
    lead_time_std: float | None = None,
    service_level: float | None = None,
    z_score: float | None = None,
    demand_period: str | None = None,
) -> dict[str, Any]:
    """Calculate Safety Stock (SS) and Reorder Point (ROP) under continuous review."""
    d = _validate_positive("avg_demand", avg_demand)
    L = _validate_positive("lead_time", lead_time)

    sigma_d = _validate_non_negative("demand_std", demand_std) if demand_std is not None else 0.0
    sigma_L = _validate_non_negative("lead_time_std", lead_time_std) if lead_time_std is not None else 0.0
    unit = demand_period or "period"

    assumptions: list[str] = [
        "Continuous review (s, Q) inventory policy where inventory position is tracked in real time.",
        "Average demand and lead time are expressed in matching time units.",
    ]

    expected_lt_demand = d * L

    variable_definitions: dict[str, str] = {
        "d": f"Average demand per {unit} (units/{unit})",
        "L": f"Average replenishment lead time ({unit}s)",
        "mu_LT": "Expected demand during lead time (units)",
        "sigma_LT": "Standard deviation of demand during lead time (units)",
        "SS": "Safety stock (units)",
        "ROP": "Reorder point (units)",
    }

    formulas: list[str] = ["mu_LT = d * L"]
    substituted_formulas: list[str] = [
        f"mu_LT = {_fmt(d)} * {_fmt(L)} = {expected_lt_demand:,.2f} units"
    ]

    if sigma_d == 0.0 and sigma_L == 0.0:
        variability_case = "constant_demand_constant_lead_time"
        assumptions.append("Both demand and lead time are deterministic (constant); zero safety stock is required.")
        sl: float | None = None
        z: float | None = None
        if service_level is not None or z_score is not None:
            sl, z, notes = _resolve_service_level_and_z(service_level=service_level, z_score=z_score)
            assumptions.extend(notes)
        variance_lt = 0.0
        sigma_lt = 0.0
        safety_stock = 0.0
        formulas.extend(["sigma_LT = 0", "SS = 0", "ROP = d * L"])
        substituted_formulas.extend(
            [
                "sigma_LT = 0.00 units (deterministic demand and lead time)",
                "SS = 0.00 units",
                f"ROP = {_fmt(d)} * {_fmt(L)} = {expected_lt_demand:,.2f} units",
            ]
        )
    else:
        sl, z, notes = _resolve_service_level_and_z(service_level=service_level, z_score=z_score)
        assumptions.extend(notes)
        variable_definitions["CSL"] = "Target cycle service level (in-stock probability per cycle)"
        variable_definitions["z"] = "Standard normal critical value = Phi^-1(CSL)"

        if sigma_d > 0.0 and sigma_L == 0.0:
            variability_case = "variable_demand_constant_lead_time"
            variable_definitions["sigma_d"] = f"Standard deviation of demand per {unit} (units/{unit})"
            assumptions.append(
                "Per-period demand is i.i.d. normal and lead time is constant; sigma_LT = sigma_d * sqrt(L)."
            )
            variance_lt = L * (sigma_d ** 2)
            sigma_lt = sigma_d * math.sqrt(L)
            formulas.append("sigma_LT = sigma_d * sqrt(L)")
            substituted_formulas.append(
                f"sigma_LT = {_fmt(sigma_d)} * sqrt({_fmt(L)}) = {sigma_lt:,.4f} units"
            )
        elif sigma_d == 0.0 and sigma_L > 0.0:
            variability_case = "constant_demand_variable_lead_time"
            variable_definitions["sigma_L"] = f"Standard deviation of lead time ({unit}s)"
            assumptions.append(
                "Demand per period is constant and lead time is normally distributed; sigma_LT = d * sigma_L."
            )
            variance_lt = (d ** 2) * (sigma_L ** 2)
            sigma_lt = d * sigma_L
            formulas.append("sigma_LT = d * sigma_L")
            substituted_formulas.append(
                f"sigma_LT = {_fmt(d)} * {_fmt(sigma_L)} = {sigma_lt:,.4f} units"
            )
        else:
            variability_case = "variable_demand_variable_lead_time"
            variable_definitions["sigma_d"] = f"Standard deviation of demand per {unit} (units/{unit})"
            variable_definitions["sigma_L"] = f"Standard deviation of lead time ({unit}s)"
            assumptions.append(
                "Demand and lead time are independent random variables; demand during lead time is approximated "
                "as normal with sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)."
            )
            term1 = L * (sigma_d ** 2)
            term2 = (d ** 2) * (sigma_L ** 2)
            variance_lt = term1 + term2
            sigma_lt = math.sqrt(variance_lt)
            formulas.append("sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)")
            substituted_formulas.append(
                f"sigma_LT = sqrt({_fmt(L)} * {_fmt(sigma_d)}^2 + {_fmt(d)}^2 * {_fmt(sigma_L)}^2) = sqrt({term1:,.2f} + {term2:,.2f}) = sqrt({variance_lt:,.2f}) = {sigma_lt:,.4f} units"
            )

        assert z is not None
        safety_stock = z * sigma_lt
        reorder_point = expected_lt_demand + safety_stock
        formulas.extend(["z = Phi^-1(CSL)", "SS = z * sigma_LT", "ROP = d * L + SS"])
        substituted_formulas.extend(
            [
                f"z = Phi^-1({sl:.4f}) = {z:.4f}",
                f"SS = {z:.4f} * {sigma_lt:,.4f} = {safety_stock:,.2f} units",
                f"ROP = {expected_lt_demand:,.2f} + {safety_stock:,.2f} = {reorder_point:,.2f} units",
            ]
        )

    reorder_point = expected_lt_demand + safety_stock

    inputs: dict[str, Any] = {
        "avg_demand": d,
        "lead_time": L,
    }
    if demand_std is not None:
        inputs["demand_std"] = sigma_d
    if lead_time_std is not None:
        inputs["lead_time_std"] = sigma_L
    if sl is not None:
        inputs["service_level"] = sl
    if z is not None:
        inputs["z_score"] = z
    if demand_period is not None:
        inputs["demand_period"] = demand_period

    intermediate: dict[str, Any] = {
        "variability_case": variability_case,
        "expected_lead_time_demand": expected_lt_demand,
        "lead_time_demand_variance": variance_lt,
        "sigma_lt": sigma_lt,
        "service_level": sl,
        "stockout_probability": (1.0 - sl) if sl is not None else 0.0,
        "z_score": z,
    }

    final_answer: dict[str, Any] = {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "recommended_reorder_point": math.ceil(reorder_point),
    }

    result: dict[str, Any] = {
        "model": "safety_stock_rop",
        "inputs": inputs,
        "variable_definitions": variable_definitions,
        "formulas": formulas,
        "substituted_formulas": substituted_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
    }
    result["verification_checks"] = verify_calculation("safety_stock_rop", result)
    return result


# ---------------------------------------------------------------------------
# 4. Fixed-Period Review (Periodic Review / Order-Up-To)
# ---------------------------------------------------------------------------


def calculate_periodic_review(
    avg_demand: float,
    review_period: float,
    lead_time: float,
    demand_std: float | None = None,
    lead_time_std: float | None = None,
    service_level: float | None = None,
    z_score: float | None = None,
    safety_stock: float | None = None,
    inventory_position: float | None = None,
    inventory_on_hand: float | None = None,
    scheduled_receipts: float | None = None,
    on_order: float | None = None,
    backorders: float | None = None,
    demand_period: str | None = None,
) -> dict[str, Any]:
    """Calculate Fixed-Period Review (Order-Up-To) parameters and replenishment order quantity."""
    d = _validate_positive("avg_demand", avg_demand)
    T = _validate_positive("review_period", review_period)
    L = _validate_non_negative("lead_time", lead_time)
    unit = demand_period or "period"

    sigma_d = _validate_non_negative("demand_std", demand_std) if demand_std is not None else 0.0
    sigma_L = _validate_non_negative("lead_time_std", lead_time_std) if lead_time_std is not None else 0.0

    if scheduled_receipts is not None and on_order is not None:
        raise ValueError("Provide either 'scheduled_receipts' OR 'on_order', not both.")
    resolved_on_order = on_order if on_order is not None else scheduled_receipts

    if inventory_position is not None and any(
        v is not None for v in (inventory_on_hand, resolved_on_order, backorders)
    ):
        raise ValueError(
            "Provide either 'inventory_position' directly OR 'inventory_on_hand' "
            "(with optional 'on_order'/'scheduled_receipts'/'backorders'), not both."
        )
    if inventory_on_hand is None and (resolved_on_order is not None or backorders is not None):
        raise ValueError(
            "'on_order'/'scheduled_receipts' or 'backorders' cannot be used without 'inventory_on_hand'."
        )

    assumptions: list[str] = [
        "Fixed-period (periodic review / order-up-to) policy where inventory is reviewed every T periods.",
        "Average demand, review period, and lead time are expressed in matching time units.",
    ]

    protection_period = T + L
    demand_during_protection = d * protection_period

    variable_definitions: dict[str, str] = {
        "d": f"Average demand per {unit} (units/{unit})",
        "T": f"Review period ({unit}s)",
        "L": f"Replenishment lead time ({unit}s)",
        "T + L": f"Protection period ({unit}s)",
        "mu_(T+L)": "Expected demand during protection period (units)",
        "SS": "Safety stock (units)",
        "Target": "Target inventory / Order-Up-To level (units)",
    }

    formulas: list[str] = [
        "Protection Period = T + L",
        "mu_(T+L) = d * (T + L)",
    ]
    substituted_formulas: list[str] = [
        f"Protection Period = {_fmt(T)} + {_fmt(L)} = {_fmt(protection_period)} {unit}s",
        f"mu_(T+L) = {_fmt(d)} * {_fmt(protection_period)} = {demand_during_protection:,.2f} units",
    ]

    sl: float | None = None
    z: float | None = None
    if safety_stock is not None:
        resolved_ss = _validate_non_negative("safety_stock", safety_stock)
        variance_protection = protection_period * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2)
        sigma_protection = math.sqrt(variance_protection)
        if service_level is not None or z_score is not None:
            sl, z, notes = _resolve_service_level_and_z(service_level=service_level, z_score=z_score)
            assumptions.extend(notes)
            if sigma_protection > 0.0 and z is not None:
                implied_ss = z * sigma_protection
                if not math.isclose(resolved_ss, implied_ss, rel_tol=1e-2, abs_tol=0.5):
                    raise ValueError(
                        f"Explicit safety_stock ({resolved_ss:.2f}) conflicts with z*sigma_(T+L) ({implied_ss:.2f})."
                    )
        assumptions.append(f"Using explicitly provided safety stock of {resolved_ss:,.2f} units.")
        formulas.append("SS = explicitly provided safety stock")
        substituted_formulas.append(f"SS = {resolved_ss:,.2f} units (explicitly given)")
    elif sigma_d == 0.0 and sigma_L == 0.0:
        if service_level is not None or z_score is not None:
            sl, z, notes = _resolve_service_level_and_z(service_level=service_level, z_score=z_score)
            assumptions.extend(notes)
        variance_protection = 0.0
        sigma_protection = 0.0
        resolved_ss = 0.0
        assumptions.append("Demand and lead time are deterministic; zero safety stock is required.")
        formulas.extend(["sigma_(T+L) = 0", "SS = 0"])
        substituted_formulas.extend(
            ["sigma_(T+L) = 0.00 units (deterministic)", "SS = 0.00 units"]
        )
    else:
        sl, z, notes = _resolve_service_level_and_z(service_level=service_level, z_score=z_score)
        assumptions.extend(notes)
        variable_definitions["CSL"] = "Target cycle service level (in-stock probability)"
        variable_definitions["z"] = "Standard normal critical value = Phi^-1(CSL)"
        variable_definitions["sigma_(T+L)"] = "Standard deviation of demand during protection period (units)"

        variance_protection = protection_period * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2)
        sigma_protection = math.sqrt(variance_protection)
        assert z is not None
        resolved_ss = z * sigma_protection
        if sigma_L == 0.0:
            variable_definitions["sigma_d"] = f"Standard deviation of demand per {unit} (units/{unit})"
            assumptions.append(
                "Per-period demand is i.i.d. normal with constant lead time; "
                "sigma_(T+L) = sigma_d * sqrt(T + L)."
            )
            formulas.extend(["sigma_(T+L) = sigma_d * sqrt(T + L)", "z = Phi^-1(CSL)", "SS = z * sigma_(T+L)"])
            substituted_formulas.extend(
                [
                    f"sigma_(T+L) = {_fmt(sigma_d)} * sqrt({_fmt(protection_period)}) = {sigma_protection:,.4f} units",
                    f"z = Phi^-1({sl:.4f}) = {z:.4f}",
                    f"SS = {z:.4f} * {sigma_protection:,.4f} = {resolved_ss:,.2f} units",
                ]
            )
        else:
            variable_definitions["sigma_d"] = f"Standard deviation of demand per {unit} (units/{unit})"
            variable_definitions["sigma_L"] = f"Standard deviation of lead time ({unit}s)"
            assumptions.append(
                "Demand and lead time are independent random variables; "
                "sigma_(T+L) = sqrt((T + L) * sigma_d^2 + d^2 * sigma_L^2)."
            )
            term1 = protection_period * (sigma_d ** 2)
            term2 = (d ** 2) * (sigma_L ** 2)
            formulas.extend(
                [
                    "sigma_(T+L) = sqrt((T + L) * sigma_d^2 + d^2 * sigma_L^2)",
                    "z = Phi^-1(CSL)",
                    "SS = z * sigma_(T+L)",
                ]
            )
            substituted_formulas.extend(
                [
                    f"sigma_(T+L) = sqrt({_fmt(protection_period)} * {_fmt(sigma_d)}^2 + {_fmt(d)}^2 * {_fmt(sigma_L)}^2) = sqrt({term1:,.2f} + {term2:,.2f}) = {sigma_protection:,.4f} units",
                    f"z = Phi^-1({sl:.4f}) = {z:.4f}",
                    f"SS = {z:.4f} * {sigma_protection:,.4f} = {resolved_ss:,.2f} units",
                ]
            )

    target_inventory = demand_during_protection + resolved_ss
    formulas.append("Target = mu_(T+L) + SS")
    substituted_formulas.append(
        f"Target = {demand_during_protection:,.2f} + {resolved_ss:,.2f} = {target_inventory:,.2f} units"
    )

    resolved_ip: float | None = None
    order_quantity: float | None = None
    recommended_order_qty: int | None = None

    if inventory_position is not None:
        resolved_ip = _ensure_finite_number("inventory_position", inventory_position)
        order_quantity = max(0.0, target_inventory - resolved_ip)
        recommended_order_qty = math.ceil(order_quantity) if order_quantity > 0 else 0
        variable_definitions["IP"] = "Current inventory position (units)"
        variable_definitions["Q"] = "Replenishment order quantity (units)"
        formulas.append("Q = max(0, Target - IP)")
        substituted_formulas.append(
            f"Q = max(0, {target_inventory:,.2f} - {resolved_ip:,.2f}) = {order_quantity:,.2f} units"
        )
    elif inventory_on_hand is not None:
        oh = _validate_non_negative("inventory_on_hand", inventory_on_hand)
        oo = _validate_non_negative("on_order", resolved_on_order) if resolved_on_order is not None else 0.0
        bo = _validate_non_negative("backorders", backorders) if backorders is not None else 0.0
        if resolved_on_order is None and backorders is None:
            assumptions.append(
                "Inventory position set equal to inventory_on_hand (no on-order inventory or backorders specified)."
            )
        resolved_ip = oh + oo - bo
        order_quantity = max(0.0, target_inventory - resolved_ip)
        recommended_order_qty = math.ceil(order_quantity) if order_quantity > 0 else 0
        variable_definitions["OH"] = "On-hand inventory (units)"
        variable_definitions["OO"] = "On-order inventory / scheduled receipts (units)"
        variable_definitions["BO"] = "Backorders (units)"
        variable_definitions["IP"] = "Inventory position = OH + OO - BO (units)"
        variable_definitions["Q"] = "Replenishment order quantity (units)"
        formulas.extend(["IP = OH + OO - BO", "Q = max(0, Target - IP)"])
        substituted_formulas.extend(
            [
                f"IP = {_fmt(oh)} + {_fmt(oo)} - {_fmt(bo)} = {resolved_ip:,.2f} units",
                f"Q = max(0, {target_inventory:,.2f} - {resolved_ip:,.2f}) = {order_quantity:,.2f} units",
            ]
        )
    else:
        assumptions.append(
            "Inventory position was not provided; order_quantity is omitted (not invented). "
            "Order max(0, target_inventory - inventory_position) at review."
        )

    inputs: dict[str, Any] = {
        "avg_demand": d,
        "review_period": T,
        "lead_time": L,
    }
    if demand_std is not None:
        inputs["demand_std"] = sigma_d
    if lead_time_std is not None:
        inputs["lead_time_std"] = sigma_L
    if sl is not None:
        inputs["service_level"] = sl
    if z is not None:
        inputs["z_score"] = z
    if safety_stock is not None:
        inputs["safety_stock"] = resolved_ss
    if inventory_position is not None:
        inputs["inventory_position"] = resolved_ip
    if inventory_on_hand is not None:
        inputs["inventory_on_hand"] = float(inventory_on_hand)
    if resolved_on_order is not None:
        inputs["on_order"] = float(resolved_on_order)
    if backorders is not None:
        inputs["backorders"] = float(backorders)
    if demand_period is not None:
        inputs["demand_period"] = demand_period

    intermediate: dict[str, Any] = {
        "protection_period": protection_period,
        "demand_during_protection_period": demand_during_protection,
        "protection_demand_variance": variance_protection,
        "sigma_protection": sigma_protection,
        "service_level": sl,
        "stockout_probability": (1.0 - sl) if sl is not None else 0.0,
        "z_score": z,
        "safety_stock_source": "explicit" if safety_stock is not None else "computed",
        "inventory_position": resolved_ip,
    }

    final_answer: dict[str, Any] = {
        "safety_stock": resolved_ss,
        "target_inventory": target_inventory,
        "recommended_target_inventory": math.ceil(target_inventory),
        "order_quantity": order_quantity,
        "recommended_order_quantity": recommended_order_qty,
    }

    result: dict[str, Any] = {
        "model": "fixed_period_review",
        "inputs": inputs,
        "variable_definitions": variable_definitions,
        "formulas": formulas,
        "substituted_formulas": substituted_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
    }
    result["verification_checks"] = verify_calculation("fixed_period_review", result)
    return result


# ---------------------------------------------------------------------------
# 5. Newsvendor (Single-Period Inventory Model)
# ---------------------------------------------------------------------------


def calculate_newsvendor(
    selling_price: float | None = None,
    unit_cost: float | None = None,
    salvage_value: float | None = None,
    goodwill_penalty: float | None = None,
    underage_cost: float | None = None,
    overage_cost: float | None = None,
    mean_demand: float | None = None,
    demand_std: float | None = None,
    distribution: str = "normal",
    min_demand: float | None = None,
    max_demand: float | None = None,
    demand_scenarios: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Solve the Single-Period Newsvendor problem."""
    assumptions: list[str] = [
        "Single-period perishable inventory setting with one upfront ordering opportunity before demand is realized.",
        "Unsold units cannot be carried over to a future selling period.",
    ]

    g = _validate_non_negative("goodwill_penalty", goodwill_penalty) if goodwill_penalty is not None else 0.0

    p: float | None = None
    c: float | None = None
    v: float | None = None

    variable_definitions: dict[str, str] = {}
    formulas: list[str] = []
    substituted_formulas: list[str] = []

    if underage_cost is not None and overage_cost is not None:
        Cu = _validate_positive("underage_cost", underage_cost)
        Co = _validate_positive("overage_cost", overage_cost)
        cost_source = "direct_underage_overage"
        variable_definitions["Cu"] = "Underage (shortage) cost per unit ($/unit)"
        variable_definitions["Co"] = "Overage (excess) cost per unit ($/unit)"
    elif selling_price is not None and unit_cost is not None:
        p = _validate_positive("selling_price", selling_price)
        c = _validate_positive("unit_cost", unit_cost)
        if salvage_value is None:
            raise ValueError(
                "Missing required 'salvage_value'. Pass salvage_value=0 explicitly if unsold units have zero salvage value."
            )
        v = _ensure_finite_number("salvage_value", salvage_value)
        Cu = p - c + g
        Co = c - v
        if Cu <= 0.0:
            raise ValueError(
                f"Underage cost (selling_price - unit_cost + goodwill_penalty = {Cu}) must be > 0."
            )
        if Co <= 0.0:
            raise ValueError(
                f"Overage cost (unit_cost - salvage_value = {Co}) must be > 0."
            )
        cost_source = "derived_from_economics"
        variable_definitions["p"] = "Selling price per unit ($/unit)"
        variable_definitions["c"] = "Purchase / unit cost ($/unit)"
        variable_definitions["v"] = "Salvage value per unsold unit ($/unit)"
        if g > 0.0:
            variable_definitions["g"] = "Goodwill penalty per unit short ($/unit)"
            formulas.append("Cu = p - c + g")
            substituted_formulas.append(
                f"Cu = ${p:,.2f} - ${c:,.2f} + ${g:,.2f} = ${Cu:,.2f}/unit"
            )
        else:
            assumptions.append("No additional stockout goodwill penalty beyond lost profit margin (g = 0).")
            formulas.append("Cu = p - c")
            substituted_formulas.append(f"Cu = ${p:,.2f} - ${c:,.2f} = ${Cu:,.2f}/unit")
        variable_definitions["Cu"] = "Underage (shortage) cost per unit ($/unit)"
        variable_definitions["Co"] = "Overage (excess) cost per unit ($/unit)"
        formulas.append("Co = c - v")
        substituted_formulas.append(f"Co = ${c:,.2f} - ${v:,.2f} = ${Co:,.2f}/unit")
    else:
        raise ValueError(
            "Provide either ('underage_cost' and 'overage_cost') OR "
            "('selling_price', 'unit_cost', and 'salvage_value')."
        )

    critical_ratio = Cu / (Cu + Co)
    variable_definitions["CR"] = "Critical ratio / optimal service fractile"
    variable_definitions["Q*"] = "Optimal order quantity (units)"
    formulas.append("CR = Cu / (Cu + Co)")
    substituted_formulas.append(
        f"CR = ${Cu:,.2f} / (${Cu:,.2f} + ${Co:,.2f}) = ${Cu:,.2f} / ${Cu + Co:,.2f} = {critical_ratio:.4f} ({critical_ratio:.2%})"
    )

    dist_name = distribution.lower().strip()
    dist_inputs: dict[str, Any] = {"distribution": dist_name}
    dist_intermediate: dict[str, Any] = {}

    if dist_name == "normal":
        if mean_demand is None or demand_std is None:
            raise ValueError("Normal newsvendor requires both 'mean_demand' and 'demand_std'.")
        mu = _validate_positive("mean_demand", mean_demand)
        sigma = _validate_positive("demand_std", demand_std)
        z_star = _STD_NORMAL.inv_cdf(critical_ratio)
        optimal_q = mu + z_star * sigma
        loss_z = _STD_NORMAL.pdf(z_star) - z_star * (1.0 - _STD_NORMAL.cdf(z_star))
        expected_shortage = sigma * loss_z
        expected_sales = mu - expected_shortage
        expected_leftover = optimal_q - expected_sales

        dist_inputs.update({"mean_demand": mu, "demand_std": sigma})
        dist_intermediate.update(
            {
                "mean_demand": mu,
                "demand_std": sigma,
                "z_score": z_star,
                "standard_normal_loss": loss_z,
            }
        )
        variable_definitions["mu"] = "Mean demand (units)"
        variable_definitions["sigma"] = "Standard deviation of demand (units)"
        variable_definitions["z*"] = "Standard normal quantile = Phi^-1(CR)"
        variable_definitions["L(z*)"] = "Standard normal loss function = phi(z*) - z* * (1 - Phi(z*))"
        formulas.extend(
            [
                "z* = Phi^-1(CR)",
                "Q* = mu + z* * sigma",
                "E[Shortage] = sigma * L(z*)",
                "E[Sales] = mu - E[Shortage]",
                "E[Leftover] = Q* - E[Sales]",
            ]
        )
        substituted_formulas.extend(
            [
                f"z* = Phi^-1({critical_ratio:.4f}) = {z_star:.4f}",
                f"Q* = {_fmt(mu)} + ({z_star:.4f}) * {_fmt(sigma)} = {optimal_q:,.2f} units",
                f"E[Shortage] = {_fmt(sigma)} * {loss_z:.4f} = {expected_shortage:,.2f} units",
                f"E[Sales] = {_fmt(mu)} - {expected_shortage:,.2f} = {expected_sales:,.2f} units",
                f"E[Leftover] = {optimal_q:,.2f} - {expected_sales:,.2f} = {expected_leftover:,.2f} units",
            ]
        )
        assumptions.append(f"Demand follows a Normal distribution N(mu={mu:g}, sigma={sigma:g}).")

    elif dist_name == "uniform":
        if min_demand is None or max_demand is None:
            raise ValueError("Uniform newsvendor requires both 'min_demand' and 'max_demand'.")
        a = _validate_non_negative("min_demand", min_demand)
        b = _validate_positive("max_demand", max_demand)
        if b <= a:
            raise ValueError(f"'max_demand' ({b}) must be strictly greater than 'min_demand' ({a}).")
        optimal_q = a + critical_ratio * (b - a)
        mu = (a + b) / 2.0
        expected_shortage = ((b - optimal_q) ** 2) / (2.0 * (b - a))
        expected_leftover = ((optimal_q - a) ** 2) / (2.0 * (b - a))
        expected_sales = mu - expected_shortage

        dist_inputs.update({"min_demand": a, "max_demand": b})
        dist_intermediate.update({"mean_demand": mu, "z_score": None})
        variable_definitions["a"] = "Minimum uniform demand (units)"
        variable_definitions["b"] = "Maximum uniform demand (units)"
        formulas.extend(
            [
                "Q* = a + CR * (b - a)",
                "E[Shortage] = (b - Q*)^2 / (2 * (b - a))",
                "E[Sales] = (a + b) / 2 - E[Shortage]",
                "E[Leftover] = (Q* - a)^2 / (2 * (b - a))",
            ]
        )
        substituted_formulas.extend(
            [
                f"Q* = {_fmt(a)} + {critical_ratio:.4f} * ({_fmt(b)} - {_fmt(a)}) = {optimal_q:,.2f} units",
                f"E[Shortage] = ({_fmt(b)} - {optimal_q:,.2f})^2 / (2 * ({_fmt(b)} - {_fmt(a)})) = {expected_shortage:,.2f} units",
                f"E[Sales] = {mu:,.2f} - {expected_shortage:,.2f} = {expected_sales:,.2f} units",
                f"E[Leftover] = ({optimal_q:,.2f} - {_fmt(a)})^2 / (2 * ({_fmt(b)} - {_fmt(a)})) = {expected_leftover:,.2f} units",
            ]
        )
        assumptions.append(f"Demand follows a Uniform distribution U[{a:g}, {b:g}].")

    elif dist_name == "discrete":
        if not demand_scenarios or not isinstance(demand_scenarios, list):
            raise ValueError("Discrete newsvendor requires a non-empty list of (demand, probability) pairs.")
        parsed_scenarios: list[tuple[float, float]] = []
        for idx, pair in enumerate(demand_scenarios):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"demand_scenarios[{idx}] must be a (demand, probability) pair.")
            d_val = _validate_non_negative(f"demand_scenarios[{idx}].demand", pair[0])
            prob = _validate_non_negative(f"demand_scenarios[{idx}].probability", pair[1])
            parsed_scenarios.append((d_val, prob))

        parsed_scenarios.sort(key=lambda x: x[0])
        total_prob = sum(prob for _, prob in parsed_scenarios)
        if not math.isclose(total_prob, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Discrete scenario probabilities must sum to 1.0 (got {total_prob}).")

        cum_prob = 0.0
        optimal_q = parsed_scenarios[-1][0]
        cum_trace: list[str] = []
        selected = False
        for d_val, prob in parsed_scenarios:
            cum_prob += prob
            cum_trace.append(f"F({_fmt(d_val)})={cum_prob:.4f}")
            if not selected and cum_prob + 1e-12 >= critical_ratio:
                optimal_q = d_val
                selected = True

        mu = sum(d_val * prob for d_val, prob in parsed_scenarios)
        expected_sales = sum(min(optimal_q, d_val) * prob for d_val, prob in parsed_scenarios)
        expected_shortage = sum(max(0.0, d_val - optimal_q) * prob for d_val, prob in parsed_scenarios)
        expected_leftover = sum(max(0.0, optimal_q - d_val) * prob for d_val, prob in parsed_scenarios)

        dist_inputs["demand_scenarios"] = parsed_scenarios
        dist_intermediate.update({"mean_demand": mu, "z_score": None})
        variable_definitions["F(d)"] = "Cumulative probability P(Demand <= d)"
        formulas.append("Q* = smallest demand d such that F(d) >= CR")
        substituted_formulas.extend(
            [
                f"Cumulative CDF: {', '.join(cum_trace)} -> first F(d) >= {critical_ratio:.4f} is Q* = {optimal_q:,.2f} units",
                f"E[Sales] = {expected_sales:,.2f} units, E[Shortage] = {expected_shortage:,.2f} units, E[Leftover] = {expected_leftover:,.2f} units",
            ]
        )
        assumptions.append("Demand follows the supplied discrete empirical probability distribution.")
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'. Use 'normal', 'uniform', or 'discrete'.")

    expected_mismatch_cost = Cu * expected_shortage + Co * expected_leftover
    formulas.append("E[Mismatch Cost] = Cu * E[Shortage] + Co * E[Leftover]")
    substituted_formulas.append(
        f"E[Mismatch Cost] = ${Cu:,.2f} * {expected_shortage:,.2f} + ${Co:,.2f} * {expected_leftover:,.2f} = ${expected_mismatch_cost:,.2f}"
    )

    expected_profit: float | None = None
    if p is not None and c is not None and v is not None:
        expected_profit = (
            p * expected_sales
            + v * expected_leftover
            - c * optimal_q
            - g * expected_shortage
        )
        if g > 0.0:
            formulas.append("E[Profit] = p * E[Sales] + v * E[Leftover] - c * Q* - g * E[Shortage]")
            substituted_formulas.append(
                f"E[Profit] = ${p:,.2f} * {expected_sales:,.2f} + ${v:,.2f} * {expected_leftover:,.2f} - ${c:,.2f} * {optimal_q:,.2f} - ${g:,.2f} * {expected_shortage:,.2f} = ${expected_profit:,.2f}"
            )
        else:
            formulas.append("E[Profit] = p * E[Sales] + v * E[Leftover] - c * Q*")
            substituted_formulas.append(
                f"E[Profit] = ${p:,.2f} * {expected_sales:,.2f} + ${v:,.2f} * {expected_leftover:,.2f} - ${c:,.2f} * {optimal_q:,.2f} = ${expected_profit:,.2f}"
            )

    inputs: dict[str, Any] = {**dist_inputs}
    if p is not None:
        inputs["selling_price"] = p
    if c is not None:
        inputs["unit_cost"] = c
    if v is not None:
        inputs["salvage_value"] = v
    if goodwill_penalty is not None:
        inputs["goodwill_penalty"] = g
    if underage_cost is not None:
        inputs["underage_cost"] = Cu
    if overage_cost is not None:
        inputs["overage_cost"] = Co

    intermediate: dict[str, Any] = {
        "cost_source": cost_source,
        "underage_cost": Cu,
        "overage_cost": Co,
        "critical_ratio": critical_ratio,
        **dist_intermediate,
        "expected_sales": expected_sales,
        "expected_shortage": expected_shortage,
        "expected_leftover": expected_leftover,
        "expected_mismatch_cost": expected_mismatch_cost,
    }
    if expected_profit is not None:
        intermediate["expected_profit"] = expected_profit

    final_answer: dict[str, Any] = {
        "underage_cost": Cu,
        "overage_cost": Co,
        "critical_ratio": critical_ratio,
        "optimal_order_quantity": optimal_q,
        "recommended_order_quantity": round(optimal_q),
        "expected_profit": expected_profit,
    }

    result: dict[str, Any] = {
        "model": "newsvendor",
        "inputs": inputs,
        "variable_definitions": variable_definitions,
        "formulas": formulas,
        "substituted_formulas": substituted_formulas,
        "intermediate_calculations": intermediate,
        "final_answer": final_answer,
        "assumptions": assumptions,
    }
    result["verification_checks"] = verify_calculation("newsvendor", result)
    return result


# ---------------------------------------------------------------------------
# Independent Verification Suite
# ---------------------------------------------------------------------------


def verify_calculation(model: str, result: dict[str, Any]) -> dict[str, Any]:
    """Independently verify mathematical invariants and sanity checks for a tool result."""
    checks: list[dict[str, Any]] = []

    def _add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    inp = result["inputs"]
    inter = result["intermediate_calculations"]
    ans = result["final_answer"]

    if model == "eoq":
        D, S, H = inp["annual_demand"], inp["ordering_cost"], inp["holding_cost"]
        Q = ans["eoq"]
        _add_check("positive_finite_eoq", Q > 0 and math.isfinite(Q), f"EOQ = {Q:.4f}")
        _add_check(
            "ordering_equals_holding_cost",
            math.isclose(inter["annual_ordering_cost"], inter["annual_holding_cost"], rel_tol=1e-6),
            f"Ordering (${inter['annual_ordering_cost']:,.2f}) vs Holding (${inter['annual_holding_cost']:,.2f})",
        )
        closed_form_tc = math.sqrt(2.0 * D * S * H)
        _add_check(
            "closed_form_total_inventory_cost",
            math.isclose(ans["total_inventory_cost"], closed_form_tc, rel_tol=1e-6),
            f"TC_inv (${ans['total_inventory_cost']:,.2f}) vs sqrt(2DSH) (${closed_form_tc:,.2f})",
        )
        tc_low = (D / (0.99 * Q)) * S + (0.99 * Q / 2.0) * H
        tc_high = (D / (1.01 * Q)) * S + (1.01 * Q / 2.0) * H
        _add_check(
            "local_optimality_convexity",
            ans["total_inventory_cost"] <= tc_low and ans["total_inventory_cost"] <= tc_high,
            f"TC(Q*)={ans['total_inventory_cost']:,.2f} <= TC(0.99Q*)={tc_low:,.2f}, TC(1.01Q*)={tc_high:,.2f}",
        )

    elif model == "quantity_discount":
        D, S = inp["annual_demand"], inp["ordering_cost"]
        Q = ans["optimal_order_quantity"]
        C = ans["optimal_unit_price"]
        H = ans["optimal_holding_cost"]
        best_tier = next(t for t in inter["tier_evaluations"] if t["tier"] == ans["optimal_tier"])

        in_bounds = (
            Q >= best_tier["min_qty"]
            and (best_tier["max_qty"] is None or Q <= best_tier["max_qty"])
            and (best_tier["upper_bound_exclusive"] is None or Q < best_tier["upper_bound_exclusive"])
        )
        _add_check(
            "optimal_qty_within_tier_bounds",
            in_bounds,
            f"Q*={Q:.2f} within tier {best_tier['tier']} bounds",
        )
        recomputed_tc = D * C + (D / Q) * S + (Q / 2.0) * H
        _add_check(
            "total_annual_cost_identity",
            math.isclose(ans["total_annual_cost"], recomputed_tc, rel_tol=1e-6),
            f"Reported (${ans['total_annual_cost']:,.2f}) vs Recomputed (${recomputed_tc:,.2f})",
        )
        feasible_costs = [t["total_annual_cost"] for t in inter["tier_evaluations"] if t["is_feasible"]]
        _add_check(
            "global_minimum_across_feasible_tiers",
            math.isclose(ans["total_annual_cost"], min(feasible_costs), rel_tol=1e-9),
            f"Chosen (${ans['total_annual_cost']:,.2f}) == min feasible (${min(feasible_costs):,.2f})",
        )

    elif model == "safety_stock_rop":
        d = inp["avg_demand"]
        L = inp["lead_time"]
        sigma_d = inp.get("demand_std", 0.0)
        sigma_L = inp.get("lead_time_std", 0.0)
        recomputed_sigma = math.sqrt(L * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2))
        _add_check(
            "sigma_lt_identity",
            math.isclose(inter["sigma_lt"], recomputed_sigma, rel_tol=1e-6, abs_tol=1e-9),
            f"sigma_LT={inter['sigma_lt']:.4f} vs {recomputed_sigma:.4f}",
        )
        if inter["service_level"] is not None and inter["z_score"] is not None:
            recomputed_z = _STD_NORMAL.inv_cdf(inter["service_level"])
            _add_check(
                "z_score_matches_service_level",
                math.isclose(inter["z_score"], recomputed_z, rel_tol=1e-2, abs_tol=0.02),
                f"z={inter['z_score']:.4f} vs inv_cdf({inter['service_level']:.4f})={recomputed_z:.4f}",
            )
        recomputed_rop = inter["expected_lead_time_demand"] + ans["safety_stock"]
        _add_check(
            "rop_equals_expected_plus_safety_stock",
            math.isclose(ans["reorder_point"], recomputed_rop, rel_tol=1e-6, abs_tol=1e-9),
            f"ROP={ans['reorder_point']:.4f} vs d*L+SS={recomputed_rop:.4f}",
        )

    elif model == "fixed_period_review":
        expected_pp = inp["review_period"] + inp["lead_time"]
        _add_check(
            "protection_period_identity",
            math.isclose(inter["protection_period"], expected_pp, rel_tol=1e-9),
            f"T + L = {inter['protection_period']:.4f} vs {expected_pp:.4f}",
        )
        recomputed_target = inter["demand_during_protection_period"] + ans["safety_stock"]
        _add_check(
            "target_inventory_identity",
            math.isclose(ans["target_inventory"], recomputed_target, rel_tol=1e-6, abs_tol=1e-9),
            f"Target={ans['target_inventory']:.4f} vs d*(T+L)+SS={recomputed_target:.4f}",
        )
        if inter["inventory_position"] is not None:
            expected_q = max(0.0, ans["target_inventory"] - inter["inventory_position"])
            _add_check(
                "order_quantity_identity",
                math.isclose(ans["order_quantity"], expected_q, rel_tol=1e-6, abs_tol=1e-9),
                f"Q={ans['order_quantity']:.4f} vs max(0, Target - IP)={expected_q:.4f}",
            )

    elif model == "newsvendor":
        Cu = ans["underage_cost"]
        Co = ans["overage_cost"]
        cr = ans["critical_ratio"]
        _add_check(
            "critical_ratio_bounds_and_formula",
            0.0 < cr < 1.0 and math.isclose(cr, Cu / (Cu + Co), rel_tol=1e-9),
            f"CR={cr:.4f} == Cu/(Cu+Co)={Cu / (Cu + Co):.4f}",
        )
        q_star = ans["optimal_order_quantity"]
        flow_q = inter["expected_sales"] + inter["expected_leftover"]
        _add_check(
            "supply_conservation_identity",
            math.isclose(flow_q, q_star, rel_tol=1e-5, abs_tol=1e-5),
            f"E[sales]+E[leftover]={flow_q:.4f} vs Q*={q_star:.4f}",
        )
        mean_d = inter["mean_demand"]
        flow_d = inter["expected_sales"] + inter["expected_shortage"]
        _add_check(
            "demand_conservation_identity",
            math.isclose(flow_d, mean_d, rel_tol=1e-5, abs_tol=1e-5),
            f"E[sales]+E[shortage]={flow_d:.4f} vs E[D]={mean_d:.4f}",
        )
        if ans["expected_profit"] is not None and "selling_price" in inp and "unit_cost" in inp:
            perfect_info_bound = (inp["selling_price"] - inp["unit_cost"]) * mean_d
            _add_check(
                "profit_below_perfect_information_bound",
                ans["expected_profit"] <= perfect_info_bound + 1e-6,
                f"E[Profit]=${ans['expected_profit']:,.2f} <= Perfect-Info Bound=${perfect_info_bound:,.2f}",
            )
    else:
        raise ValueError(f"Unknown model for verification: {model}")

    return {
        "verified": all(c["passed"] for c in checks),
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "total_checks": len(checks),
        "checks": checks,
    }


# Backward-compatible aliases
calculate_safety_stock_rop = calculate_rop
calculate_fixed_period_review = calculate_periodic_review


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "eoq": {
        "display_name": "Economic Order Quantity (EOQ)",
        "tool_name": "calculate_eoq",
        "function": calculate_eoq,
        "required_inputs": [
            "annual_demand",
            "ordering_cost",
            "holding_cost (or unit_cost + holding_cost_rate)",
        ],
    },
    "quantity_discount": {
        "display_name": "Quantity Discount (All-Units EOQ)",
        "tool_name": "calculate_quantity_discount",
        "function": calculate_quantity_discount,
        "required_inputs": [
            "annual_demand",
            "ordering_cost",
            "price_breaks",
            "holding_cost_rate (or fixed holding_cost)",
        ],
    },
    "safety_stock_rop": {
        "display_name": "Safety Stock and Reorder Point",
        "tool_name": "calculate_rop",
        "function": calculate_rop,
        "required_inputs": [
            "avg_demand",
            "lead_time",
            "service_level or z_score (required when demand_std or lead_time_std > 0)",
        ],
    },
    "fixed_period_review": {
        "display_name": "Fixed-Period Review (Order-Up-To)",
        "tool_name": "calculate_periodic_review",
        "function": calculate_periodic_review,
        "required_inputs": [
            "avg_demand",
            "review_period",
            "lead_time",
            "service_level or z_score or explicit safety_stock",
        ],
    },
    "newsvendor": {
        "display_name": "Newsvendor (Single-Period Model)",
        "tool_name": "calculate_newsvendor",
        "function": calculate_newsvendor,
        "required_inputs": [
            "selling_price, unit_cost, salvage_value (or underage_cost & overage_cost)",
            "mean_demand & demand_std (or uniform/discrete demand)",
        ],
    },
}
