"""Provider signal calculators (providers only — never touches claim logic).

Deterministic Analytical Engine signal calculations:

    input attributes -> deterministic calculation -> supporting signal

Every signal is traceable to actual submitted provider attributes and/or the
training medians stored inside the provider artifact. Signals are 0-100
evidence strengths; they are NOT a second fraud score and they never modify
the ML prediction or the ML risk score.

Categories covered: Financial, Utilization, Service concentration,
Peer deviation, Behavioral/provider activity.
"""
from __future__ import annotations

from typing import Any

from ..state import SignalRecord
from . import (
    abs_deviation,
    clip,
    downside_deviation,
    fmt,
    fmt_money,
    make_signal,
    median_of,
    num,
    upside_deviation,
)


def calculate_provider_signals(row: dict[str, Any], medians: dict[str, float]) -> list[SignalRecord]:
    signals: list[SignalRecord] = []

    # ------------------------------------------------------ financial behavior
    # 1. Charge aggressiveness — submitted charges above the peer-relevant median
    #    combined with a low payment-to-charge ratio.
    sub = num(row, "cms_weighted_avg_submitted_charge")
    sub_med = median_of(medians, "cms_weighted_avg_submitted_charge", 0.0)
    ptc = num(row, "cms_payment_to_charge_ratio")
    ptc_med = median_of(medians, "cms_payment_to_charge_ratio", 0.27)
    value = 0.5 * upside_deviation(sub or 0.0, sub_med) + 0.5 * downside_deviation(ptc or ptc_med, ptc_med)
    signals.append(make_signal(
        "Charge Aggressiveness", value,
        f"Weighted avg submitted charge {fmt_money(sub)} vs training median {fmt_money(sub_med)}; "
        f"payment-to-charge ratio {fmt(ptc)} vs {fmt(ptc_med)}.",
        ["cms_weighted_avg_submitted_charge", "cms_payment_to_charge_ratio"], row,
    ))

    # 2. Reimbursement ratio anomaly — payment vs charge / allowed / standardized.
    parts = [
        abs_deviation(v, median_of(medians, col, fb))
        for col, fb, v in (
            ("cms_payment_to_charge_ratio", 0.27, ptc),
            ("cms_payment_to_allowed_ratio", 0.85, num(row, "cms_payment_to_allowed_ratio")),
            ("cms_payment_to_standardized_ratio", 1.0, num(row, "cms_payment_to_standardized_ratio")),
        )
        if v is not None
    ]
    value = max(parts) if parts else 0.0
    signals.append(make_signal(
        "Reimbursement Ratio Anomaly", value,
        f"Payment-to-charge {fmt(ptc)}, payment-to-allowed "
        f"{fmt(num(row, 'cms_payment_to_allowed_ratio'))}, payment-to-standardized "
        f"{fmt(num(row, 'cms_payment_to_standardized_ratio'))} vs training medians.",
        ["cms_payment_to_charge_ratio", "cms_payment_to_allowed_ratio", "cms_payment_to_standardized_ratio"], row,
    ))

    # 3. Allowed/standardized spread — allowed amount and standardized payment
    #    deviating from training medians (either direction).
    allowed = num(row, "cms_weighted_avg_allowed_amount")
    std = num(row, "cms_weighted_avg_standardized_payment")
    parts = [
        abs_deviation(allowed or 0.0, median_of(medians, "cms_weighted_avg_allowed_amount", 0.0)),
        abs_deviation(std or 0.0, median_of(medians, "cms_weighted_avg_standardized_payment", 0.0)),
    ]
    value = max(parts)
    signals.append(make_signal(
        "Allowed/Standardized Spread", value,
        f"Weighted avg allowed {fmt_money(allowed)} and standardized payment "
        f"{fmt_money(std)} vs training medians.",
        ["cms_weighted_avg_allowed_amount", "cms_weighted_avg_standardized_payment"], row,
    ))

    # 4. Payment volume — weighted average payment above the training median.
    pay = num(row, "cms_weighted_avg_payment")
    pay_med = median_of(medians, "cms_weighted_avg_payment", 0.0)
    value = upside_deviation(pay or 0.0, pay_med)
    signals.append(make_signal(
        "Payment Volume", value,
        f"Weighted avg payment {fmt_money(pay)} vs training median {fmt_money(pay_med)}.",
        ["cms_weighted_avg_payment"], row,
    ))

    # ---------------------------------------------------- utilization behavior
    # 5. Beneficiary reach — total beneficiaries above the training median.
    benes = num(row, "cms_total_beneficiaries")
    benes_med = median_of(medians, "cms_total_beneficiaries", 1.0)
    value = upside_deviation(benes or 0.0, benes_med)
    signals.append(make_signal(
        "Beneficiary Reach", value,
        f"{fmt(benes, 0)} beneficiaries vs training median {fmt(benes_med, 0)}.",
        ["cms_total_beneficiaries"], row,
    ))

    # 6. Service volume — total services, beneficiary days and claim count
    #    above medians.
    services = num(row, "cms_total_services")
    days = num(row, "cms_total_beneficiary_days")
    claim_count = num(row, "claim_count")
    parts = [
        upside_deviation(services or 0.0, median_of(medians, "cms_total_services", 1.0)),
        upside_deviation(days or 0.0, median_of(medians, "cms_total_beneficiary_days", 0.0)),
        upside_deviation(claim_count or 0.0, median_of(medians, "claim_count", 1.0)),
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Service Volume", value,
        f"{fmt(services, 0)} services, {fmt(days, 0)} beneficiary days and "
        f"{fmt(claim_count, 0)} claims vs training medians.",
        ["cms_total_services", "cms_total_beneficiary_days", "claim_count"], row,
    ))

    # 7. Services per beneficiary — utilization intensity above the median.
    spb = num(row, "services_per_beneficiary")
    spb_med = median_of(medians, "services_per_beneficiary", 1.0)
    value = upside_deviation(spb or 0.0, spb_med)
    signals.append(make_signal(
        "Services Per Beneficiary", value,
        f"{fmt(spb)} services per beneficiary vs training median {fmt(spb_med)}.",
        ["services_per_beneficiary"], row,
    ))

    # --------------------------------------------------- service concentration
    # 8. HCPCS concentration — Herfindahl index + top-code share + top-10% share.
    hhi = num(row, "hcpcs_concentration_hhi", 0.0) or 0.0
    top_ratio = num(row, "top_hcpcs_service_ratio", 0.0) or 0.0
    top10 = num(row, "top_hcpcs_10pct_share", 0.0) or 0.0
    value = 0.5 * (hhi * 100.0) + 0.3 * (top_ratio * 100.0) + 0.2 * (top10 * 100.0)
    signals.append(make_signal(
        "HCPCS Concentration", value,
        f"HHI {fmt(hhi)}, top-code share {fmt(top_ratio)}, top-10% share {fmt(top10)}.",
        ["hcpcs_concentration_hhi", "top_hcpcs_service_ratio", "top_hcpcs_10pct_share"], row,
    ))

    # 9. Service mix narrowness — unique HCPCS / places of service / drug services
    #    BELOW training medians indicate a narrow, focused billing profile.
    parts = [
        downside_deviation(v or 0.0, median_of(medians, col, fb))
        for col, fb, v in (
            ("cms_unique_hcpcs", 1.0, num(row, "cms_unique_hcpcs")),
            ("cms_unique_places_of_service", 1.0, num(row, "cms_unique_places_of_service")),
            ("cms_drug_services", 0.0, num(row, "cms_drug_services")),
        )
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Service Mix Narrowness", value,
        f"{fmt(num(row, 'cms_unique_hcpcs'), 0)} unique HCPCS, "
        f"{fmt(num(row, 'cms_unique_places_of_service'), 0)} places of service, "
        f"{fmt(num(row, 'cms_drug_services'), 0)} drug services vs training medians.",
        ["cms_unique_hcpcs", "cms_unique_places_of_service", "cms_drug_services"], row,
    ))

    # ----------------------------------------------------------- peer deviation
    # 10. Aggregate peer deviation — the engineered peer deviation score.
    peer_dev = num(row, "peer_deviation_score", 0.0) or 0.0
    value = clip(peer_dev * 100.0)
    signals.append(make_signal(
        "Aggregate Peer Deviation", value,
        f"Engineered peer deviation score {fmt(peer_dev, 3)} "
        f"(treatment profile distance from the peer group).",
        ["peer_deviation_score"], row,
    ))

    # 11. Treatment intensity vs peers — services / payment / charge / allowed /
    #     standardized payment relative to the peer-group medians.
    cols = [
        "treatment_services_vs_peer_median",
        "treatment_payment_vs_peer_median",
        "treatment_charge_vs_peer_median",
        "treatment_allowed_vs_peer_median",
        "treatment_standardized_payment_vs_peer_median",
    ]
    parts = [upside_deviation(num(row, c, 1.0) or 1.0, 1.0) for c in cols]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Treatment Intensity vs Peers", value,
        f"Treatment metrics vs peer medians: services {fmt(num(row, cols[0]))}, "
        f"payment {fmt(num(row, cols[1]))}, charge {fmt(num(row, cols[2]))}.",
        cols, row,
    ))

    # --------------------------------------- behavioral / provider activity
    # 12. Peer percentile position — treatment service/payment percentiles.
    svc_pct = num(row, "treatment_service_percentile", 0.5) or 0.5
    pay_pct = num(row, "treatment_payment_percentile", 0.5) or 0.5
    value = 0.5 * (svc_pct * 100.0) + 0.5 * (pay_pct * 100.0)
    signals.append(make_signal(
        "Peer Percentile Position", value,
        f"Treatment service percentile P{round(svc_pct * 100)}, "
        f"payment percentile P{round(pay_pct * 100)} within the peer group.",
        ["treatment_service_percentile", "treatment_payment_percentile"], row,
    ))

    return signals
