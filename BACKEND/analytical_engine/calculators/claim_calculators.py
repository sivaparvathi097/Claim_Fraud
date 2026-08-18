"""Claim signal calculators (claims only — never touches provider logic).

Deterministic Analytical Engine signal calculations:

    input attributes -> deterministic calculation -> supporting signal

Every signal is traceable to actual submitted attributes and/or the training
medians stored inside the claim artifact. Signals are 0-100 evidence
strengths; they are NOT a second fraud score and they never modify the ML
prediction or the ML risk score.

Categories covered: Financial, Utilization, Temporal, Peer-relative,
Beneficiary/provider behavior.
"""
from __future__ import annotations

from typing import Any

from ..state import SignalRecord
from . import (
    abs_deviation,
    clip,
    fmt,
    fmt_money,
    make_signal,
    median_of,
    num,
    upside_deviation,
)


def calculate_claim_signals(row: dict[str, Any], medians: dict[str, float]) -> list[SignalRecord]:
    signals: list[SignalRecord] = []

    # ------------------------------------------------------ financial behavior
    # 1. Payment peer deviation — payment relative to the peer-cohort median
    #    plus the payment percentile inside the comparable cohort.
    vs_median = median_of(medians, "Payment_Vs_Peer_Median", 1.0)
    vs_peer = num(row, "Payment_Vs_Peer_Median", vs_median) or vs_median
    pay_pct = num(row, "Payment_Percentile", 0.5) or 0.5
    value = 0.6 * upside_deviation(vs_peer, 1.0) + 0.4 * (pay_pct * 100.0)
    signals.append(make_signal(
        "Payment Peer Deviation", value,
        f"Claim payment is {fmt(vs_peer)}x the peer-cohort median and sits at the "
        f"P{round(pay_pct * 100)} payment percentile.",
        ["Payment_Vs_Peer_Median", "Payment_Percentile"], row,
    ))

    # 2. Payment ratio anomaly — payment-to-charge and payment-to-allowed ratios
    #    vs their training medians (deviation in either direction is atypical).
    ptc = num(row, "Payment_To_Charge_Ratio")
    pta = num(row, "Payment_To_Allowed_Ratio")
    atc = num(row, "Allowed_To_Charge_Ratio")
    dev_parts: list[float] = []
    if ptc is not None:
        dev_parts.append(abs_deviation(ptc, median_of(medians, "Payment_To_Charge_Ratio", 0.3)))
    if pta is not None:
        dev_parts.append(abs_deviation(pta, median_of(medians, "Payment_To_Allowed_Ratio", 0.85)))
    if atc is not None:
        dev_parts.append(abs_deviation(atc, median_of(medians, "Allowed_To_Charge_Ratio", 0.9)))
    value = max(dev_parts) if dev_parts else 0.0
    signals.append(make_signal(
        "Payment Ratio Anomaly", value,
        f"Payment-to-charge {fmt(ptc)}, payment-to-allowed {fmt(pta)}, "
        f"allowed-to-charge {fmt(atc)} vs training medians.",
        ["Payment_To_Charge_Ratio", "Payment_To_Allowed_Ratio", "Allowed_To_Charge_Ratio"], row,
    ))

    # 3. Payment per day — daily payment rate above the training median.
    ppd = num(row, "Claim_Payment_Per_Day")
    ppd_med = median_of(medians, "Claim_Payment_Per_Day", 0.0)
    value = upside_deviation(ppd or 0.0, ppd_med)
    signals.append(make_signal(
        "Payment Per Day", value,
        f"Payment per day {fmt_money(ppd)} vs training median {fmt_money(ppd_med)}.",
        ["Claim_Payment_Per_Day"], row,
    ))

    # ---------------------------------------------------- utilization behavior
    # 4. Billing intensity — service count above median + service-count percentile.
    svc = num(row, "Service_Count")
    svc_med = median_of(medians, "Service_Count", 1.0)
    svc_pct = num(row, "Service_Count_Percentile", 0.5) or 0.5
    value = 0.5 * upside_deviation(svc or 0.0, svc_med) + 0.5 * (svc_pct * 100.0)
    signals.append(make_signal(
        "Billing Intensity", value,
        f"{fmt(svc, 0)} service lines vs training median {fmt(svc_med, 0)} "
        f"(P{round(svc_pct * 100)} percentile).",
        ["Service_Count", "Service_Count_Percentile"], row,
    ))

    # 5. Clinical coding spread — procedures / diagnoses / unique HCPCS codes
    #    above their training medians.
    proc = num(row, "Procedure_Count")
    dx = num(row, "Diagnosis_Count")
    hcpcs = num(row, "Unique_HCPCS_Count")
    parts = [
        upside_deviation(proc or 0.0, median_of(medians, "Procedure_Count", 1.0)),
        upside_deviation(dx or 0.0, median_of(medians, "Diagnosis_Count", 1.0)),
        upside_deviation(hcpcs or 0.0, median_of(medians, "Unique_HCPCS_Count", 1.0)),
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Clinical Coding Spread", value,
        f"{fmt(proc, 0)} procedures, {fmt(dx, 0)} diagnoses, {fmt(hcpcs, 0)} unique "
        f"HCPCS codes vs training medians.",
        ["Procedure_Count", "Diagnosis_Count", "Unique_HCPCS_Count"], row,
    ))

    # 6. Duration anomaly — claim duration vs median + duration percentile.
    dur = num(row, "Claim_Duration_Days")
    dur_med = median_of(medians, "Claim_Duration_Days", 1.0)
    dur_pct = num(row, "Duration_Percentile", 0.5) or 0.5
    value = 0.5 * upside_deviation(dur or 0.0, dur_med) + 0.5 * (dur_pct * 100.0)
    signals.append(make_signal(
        "Duration Anomaly", value,
        f"Claim spans {fmt(dur, 0)} days vs training median {fmt(dur_med, 0)} "
        f"(P{round(dur_pct * 100)} percentile).",
        ["Claim_Duration_Days", "Duration_Percentile"], row,
    ))

    # --------------------------------------------------------- temporal behavior
    # 7. Claim frequency burst — claims in the last 7/30/90 days above medians.
    c7 = num(row, "Claims_Last_7_Days", 0.0) or 0.0
    c30 = num(row, "Claims_Last_30_Days", 0.0) or 0.0
    c90 = num(row, "Claims_Last_90_Days", 0.0) or 0.0
    prev = num(row, "Previous_Claim_Count", 0.0) or 0.0
    parts = [
        upside_deviation(c7, median_of(medians, "Claims_Last_7_Days", 0.0)),
        upside_deviation(c30, median_of(medians, "Claims_Last_30_Days", 1.0)),
        upside_deviation(c90, median_of(medians, "Claims_Last_90_Days", 2.0)),
        upside_deviation(prev, median_of(medians, "Previous_Claim_Count", 1.0)),
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Claim Frequency Burst", value,
        f"{int(c7)} claims in 7d, {int(c30)} in 30d, {int(c90)} in 90d "
        f"({int(prev)} previous claims overall).",
        ["Claims_Last_7_Days", "Claims_Last_30_Days", "Claims_Last_90_Days", "Previous_Claim_Count"], row,
    ))

    # 8. Rapid resubmission gap — days since the previous claim relative to the
    #    training median gap (short gaps raise the signal).
    gap = num(row, "Days_Since_Previous_Claim")
    gap_med = median_of(medians, "Days_Since_Previous_Claim", 30.0)
    if gap is not None and gap_med > 0:
        value = clip((gap_med - gap) / gap_med * 100.0)
    else:
        value = 0.0
    signals.append(make_signal(
        "Rapid Resubmission", value,
        f"{fmt(gap, 0)} days since the previous claim vs median gap {fmt(gap_med, 0)} days.",
        ["Days_Since_Previous_Claim"], row,
    ))

    # 9. Recent payment volume — payment accumulated in the last 30 days.
    pay30 = num(row, "Payment_Last_30_Days", 0.0) or 0.0
    pay30_med = median_of(medians, "Payment_Last_30_Days", 0.0)
    value = upside_deviation(pay30, pay30_med)
    signals.append(make_signal(
        "Recent Payment Volume", value,
        f"{fmt_money(pay30)} paid in the last 30 days vs training median {fmt_money(pay30_med)}.",
        ["Payment_Last_30_Days"], row,
    ))

    # ---------------------------------------------- beneficiary/provider behavior
    # 10. Patient-provider concentration — repeated claims / spending between the
    #     same beneficiary and provider above training medians.
    pp_cnt = num(row, "Patient_Provider_Claim_Count")
    pp_spend = num(row, "Patient_Provider_Spending")
    parts = [
        upside_deviation(pp_cnt or 0.0, median_of(medians, "Patient_Provider_Claim_Count", 1.0)),
        upside_deviation(pp_spend or 0.0, median_of(medians, "Patient_Provider_Spending", 0.0)),
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Patient-Provider Concentration", value,
        f"{fmt(pp_cnt, 0)} claims and {fmt_money(pp_spend)} between this patient and provider "
        f"vs training medians.",
        ["Patient_Provider_Claim_Count", "Patient_Provider_Spending"], row,
    ))

    # 11. Provider volume pressure — provider claim count / beneficiary panel size.
    prov_claims = num(row, "Provider_Claim_Count")
    prov_benes = num(row, "Provider_Beneficiary_Count")
    parts = [
        upside_deviation(prov_claims or 0.0, median_of(medians, "Provider_Claim_Count", 1.0)),
        upside_deviation(prov_benes or 0.0, median_of(medians, "Provider_Beneficiary_Count", 1.0)),
    ]
    value = sum(parts) / len(parts)
    signals.append(make_signal(
        "Provider Volume Pressure", value,
        f"Provider submitted {fmt(prov_claims, 0)} claims across {fmt(prov_benes, 0)} "
        f"beneficiaries vs training medians.",
        ["Provider_Claim_Count", "Provider_Beneficiary_Count"], row,
    ))

    return signals
