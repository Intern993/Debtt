#!/usr/bin/env python3
"""
GoldenPi bond recommendation engine built to match the user's policy sheet.

Policy hierarchy implemented here:

1) Risk-score banding
   Conservative: score < 41
   Moderate:     41 <= score <= 65
   Aggressive:   score >= 66

2) Profile bucket weights
   Conservative -> C 60%, M 20%, A 20%
   Moderate     -> C 40%, M 30%, A 30%
   Aggressive   -> C 20%, M 40%, A 40%

3) Preference order within each bucket
   Single platform availability
   Time horizon matching
   Listed securities preference
   Yield optimization
   Liquidity and issuer evaluation

4) Ticket-size / bond-count bands
   10L–25L   -> 1 bond per bucket
   25L–50L   -> 1 bond per bucket (range in the sheet is 1–2; this script uses the lower bound by default)
   50L–1CR   -> 2 bonds per bucket
   1CR+      -> 2 bonds per bucket (range in the sheet is 2–3; this script uses the lower bound by default)

Important limitation:
The GoldenPi source file you shared does not contain an explicit listed/unlisted column.
The script therefore treats listed preference as neutral unless a source column is present.
If you later add a listed flag column, the same engine will use it automatically.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# -----------------------------
# Policy constants
# -----------------------------

PROFILE_POLICIES = {
    "Conservative": {
        "bucket_weights": {"C": 0.60, "M": 0.20, "A": 0.20},
        "avg_weighted_ytm": 7.40,
        "default_bucket_label": "C",
    },
    "Moderate": {
        "bucket_weights": {"C": 0.40, "M": 0.30, "A": 0.30},
        "avg_weighted_ytm": 8.98,
        "default_bucket_label": "M",
    },
    "Aggressive": {
        "bucket_weights": {"C": 0.20, "M": 0.40, "A": 0.40},
        "avg_weighted_ytm": 11.00,
        "default_bucket_label": "A",
    },
}

RISK_SCORE_TO_PROFILE = [
    ("Conservative", lambda x: x < 41),
    ("Moderate", lambda x: 41 <= x <= 65),
    ("Aggressive", lambda x: x >= 66),
]

# Bucket mapping for GoldenPi data.
# The source file mainly gives credit ratings, so the mapping is rating-driven.
# This is aligned to the user's correction that AA should not be forced into C.
def rating_to_bucket(rating_text: str) -> str:
    rating = normalize_rating(rating_text)

    if rating == "AAA":
        return "C"
    if rating == "AA":
        return "M"
    if rating in {"A+", "A", "A-", "BBB+", "BBB", "BBB-"}:
        return "A"

    # Conservative fallback: if rating cannot be parsed, keep it out of the shortlist.
    return "UNK"


def bucket_preference_for_profile(profile: str) -> list[str]:
    # Used only as a tie-break / fallback priority when a bucket has insufficient options.
    if profile == "Conservative":
        return ["C", "M", "A"]
    if profile == "Moderate":
        return ["C", "M", "A"]
    return ["C", "M", "A"]


def bonds_per_bucket_for_amount(amount_inr: float) -> int:
    """
    Lower-bound interpretation of the sheet's 'Bonds per bucket' ranges.
    The sheet shows ranges:
      10L-25L -> 1
      25L-50L -> 1-2
      50L-1CR -> 2
      1CR+    -> 2-3

    This function returns the lower bound by default.
    """
    if amount_inr < 2_500_000:
        return 1
    if amount_inr < 5_000_000:
        return 1
    if amount_inr < 10_000_000:
        return 2
    return 2


# -----------------------------
# Parsing helpers
# -----------------------------

def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_rating(rating_text: Any) -> str:
    txt = normalize_text(rating_text).upper()
    if not txt:
        return ""

    # Focus on the dominant rating token only.
    # Examples:
    #   "A- ACUITE"            -> "A-"
    #   "AA IND/ AA ACUITE"    -> "AA"
    #   "AAA CRISIL/AAA CARE"  -> "AAA"
    #   "BBB+ ICRA"            -> "BBB+"
    m = re.search(r"\b(AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB)\b", txt)
    return m.group(1) if m else txt


def parse_percent(value: Any) -> Optional[float]:
    """
    Convert coupon/YTM style values to percentage points.
    Examples:
      0.1125 -> 11.25
      11.25  -> 11.25
      "11.25%" -> 11.25
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 1:
            return v * 100.0
        return v

    txt = str(value).strip().replace("%", "")
    if not txt:
        return None
    try:
        v = float(txt)
    except ValueError:
        return None
    if v <= 1:
        return v * 100.0
    return v


def parse_inr_band(value: Any) -> Optional[float]:
    """
    Parse strings like:
      5L   -> 500000
      10L  -> 1000000
      1Cr+ -> 10000000
      2Cr+ -> 20000000
    """
    if pd.isna(value):
        return None

    txt = str(value).strip().upper().replace(",", "").replace(" ", "")
    if not txt:
        return None

    # strip trailing plus
    txt = txt.replace("+", "")

    # Common band patterns
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(L|CR)", txt)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if unit == "L":
            return num * 100_000
        return num * 10_000_000

    # Raw integer string
    try:
        return float(txt)
    except ValueError:
        return None


def parse_date(value: Any) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def infer_profile_from_score(score: float) -> str:
    for profile, predicate in RISK_SCORE_TO_PROFILE:
        if predicate(score):
            return profile
    raise ValueError(f"Could not classify risk score: {score}")


def infer_listed_flag(row: pd.Series) -> Optional[bool]:
    """
    GoldenPi source file does not include an explicit listed/unlisted column.
    If a suitable column appears later, we detect it here.
    """
    candidate_cols = [
        "Listed",
        "Listed/Unlisted",
        "Listed Status",
        "Listing Status",
        "Status",
    ]
    for col in candidate_cols:
        if col in row.index and not pd.isna(row[col]):
            txt = normalize_text(row[col]).lower()
            if txt in {"listed", "yes", "true", "y"}:
                return True
            if txt in {"unlisted", "no", "false", "n"}:
                return False
    return None


def score_from_rating(rating_text: str) -> float:
    """
    Issuer-quality proxy from the rating token only.
    """
    rating = normalize_rating(rating_text)
    mapping = {
        "AAA": 100.0,
        "AA+": 92.0,
        "AA": 86.0,
        "AA-": 80.0,
        "A+": 70.0,
        "A": 62.0,
        "A-": 54.0,
        "BBB+": 42.0,
        "BBB": 35.0,
        "BBB-": 28.0,
    }
    return mapping.get(rating, 50.0)


def parse_target_duration_months(value: Any) -> float:
    """
    Duration input is expected in months.
    """
    if value is None:
        raise ValueError("Duration is required.")
    return float(value)


def days_to_target_horizon(months: float, as_of: Optional[pd.Timestamp] = None) -> int:
    as_of = as_of or pd.Timestamp.today().normalize()
    target = as_of + pd.DateOffset(months=int(round(months)))
    return int((target - as_of).days)


def get_ticket_band(amount_inr: float) -> str:
    if amount_inr < 2_500_000:
        return "10L-25L"
    if amount_inr < 5_000_000:
        return "25L-50L"
    if amount_inr < 10_000_000:
        return "50L-1CR"
    return "1CR+"


# -----------------------------
# Data preparation
# -----------------------------

def load_goldenpi(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Reads the GoldenPi sheet. The file has non-data banner rows; header is row 5 (0-based header=4).
    """
    sheet_name = sheet_name or 0
    df = pd.read_excel(path, sheet_name=sheet_name, header=4)
    df = df.dropna(how="all")
    if "ISIN" not in df.columns:
        raise ValueError("Could not find the expected GoldenPi header row.")
    df = df.dropna(subset=["ISIN"]).copy()
    return df


def normalize_goldenpi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["isin"] = out["ISIN"].map(normalize_text)
    out["security_name"] = out["Security"].map(normalize_text)
    out["face_value"] = pd.to_numeric(out.get("Face Value"), errors="coerce")
    out["coupon_pct"] = out.get("Coupon").map(parse_percent)
    out["ytm_pct"] = out.get("YTM/YTC").map(parse_percent)
    out["maturity_date"] = out.get("Maturity Date").map(parse_date)
    out["credit_rating_raw"] = out.get("Credit Rating").map(normalize_text)
    out["credit_rating"] = out["credit_rating_raw"].map(normalize_rating)
    out["taxable"] = out.get("Taxable").map(normalize_text)
    out["last_ip_date"] = out.get("Last IP Date").map(parse_date)
    out["next_ip_date"] = out.get("Next IP Date").map(parse_date)
    out["min_lot_inr"] = out.get("Min lot").map(parse_inr_band)
    out["qtm_inr"] = out.get("QTM").map(parse_inr_band)
    out["listed"] = out.apply(infer_listed_flag, axis=1)
    out["bucket"] = out["credit_rating_raw"].map(bucket_to_bucket_with_fallback)

    # Basic cleanup
    out = out[out["isin"] != ""].copy()
    out = out[out["maturity_date"].notna()].copy()
    out = out[out["ytm_pct"].notna()].copy()
    out = out[out["min_lot_inr"].notna()].copy()
    return out


def bucket_to_bucket_with_fallback(rating_text: Any) -> str:
    bucket = rating_to_bucket(rating_text)
    return bucket


# -----------------------------
# Ranking logic
# -----------------------------

def ranked_candidates(
    df: pd.DataFrame,
    profile: str,
    amount_inr: float,
    duration_months: float,
    allow_unlisted: bool,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    as_of = as_of or pd.Timestamp.today().normalize()
    target_days = days_to_target_horizon(duration_months, as_of=as_of)

    tmp = df.copy()

    # Hard eligibility: minimum lot must fit at least one lot.
    tmp = tmp[tmp["min_lot_inr"] <= amount_inr].copy()

    # Hard eligibility: maturity must be in the future.
    tmp["days_to_maturity"] = (tmp["maturity_date"] - as_of).dt.days
    tmp = tmp[tmp["days_to_maturity"] > 0].copy()

    # Bucket eligibility: exact rating-based bucket.
    tmp = tmp[tmp["bucket"].isin(["C", "M", "A"])].copy()

    # Listed preference if data exists.
    if not allow_unlisted and tmp["listed"].notna().any():
        tmp = tmp[(tmp["listed"] == True) | (tmp["listed"].isna())].copy()

    # Preference measures
    tmp["duration_gap_days"] = (tmp["days_to_maturity"] - target_days).abs()

    # Listed priority: listed first; unknown neutral; unlisted last.
    def listed_priority(v: Any) -> int:
        if v is True:
            return 2
        if v is False:
            return 0
        return 1

    tmp["listed_priority_score"] = tmp["listed"].map(listed_priority) if "listed" in tmp.columns else 1

    # Yield score within the candidate pool (higher is better)
    ytm_min = tmp["ytm_pct"].min()
    ytm_max = tmp["ytm_pct"].max()
    if pd.isna(ytm_min) or pd.isna(ytm_max) or math.isclose(float(ytm_min), float(ytm_max)):
        tmp["yield_score"] = 50.0
    else:
        tmp["yield_score"] = (tmp["ytm_pct"] - ytm_min) / (ytm_max - ytm_min) * 100.0

    # Liquidity proxy: higher QTM and smaller minimum lot are better.
    qtm_max = tmp["qtm_inr"].max(skipna=True)
    lot_max = tmp["min_lot_inr"].max(skipna=True)

    if pd.isna(qtm_max) or qtm_max <= 0:
        tmp["qtm_score"] = 50.0
    else:
        tmp["qtm_score"] = (tmp["qtm_inr"].fillna(0) / qtm_max) * 100.0

    if pd.isna(lot_max) or lot_max <= 0:
        tmp["lot_score"] = 50.0
    else:
        tmp["lot_score"] = 100.0 - (tmp["min_lot_inr"].fillna(lot_max) / lot_max) * 100.0

    tmp["liquidity_score"] = (0.65 * tmp["qtm_score"]) + (0.35 * tmp["lot_score"])
    tmp["issuer_score"] = tmp["credit_rating"].map(score_from_rating)

    # Platform score: the source file is GoldenPi only, so every row is on one platform.
    # This keeps the hierarchy explicit and future-proof.
    tmp["platform_score"] = 100.0

    # Exact preference order as a lexicographic sort:
    # 1) platform score
    # 2) time horizon match (smaller gap)
    # 3) listed preference
    # 4) yield optimization
    # 5) liquidity
    # 6) issuer quality
    tmp = tmp.sort_values(
        by=[
            "platform_score",
            "duration_gap_days",
            "listed_priority_score",
            "yield_score",
            "liquidity_score",
            "issuer_score",
        ],
        ascending=[False, True, False, False, False, False],
        kind="mergesort",
    ).copy()

    # Profile bucket weights
    weights = PROFILE_POLICIES[profile]["bucket_weights"]
    tmp["bucket_weight"] = tmp["bucket"].map(weights).fillna(0.0)

    return tmp


def choose_bonds(
    ranked: pd.DataFrame,
    profile: str,
    amount_inr: float,
    duration_months: float,
    allow_unlisted: bool,
    as_of: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      selected allocations dataframe
      bucket summary dataframe
    """
    as_of = as_of or pd.Timestamp.today().normalize()
    target_days = days_to_target_horizon(duration_months, as_of=as_of)

    # Default number of bonds per bucket based on the sheet's banding.
    bonds_per_bucket = bonds_per_bucket_for_amount(amount_inr)

    bucket_weights = PROFILE_POLICIES[profile]["bucket_weights"]
    bucket_order = bucket_preference_for_profile(profile)

    selected_rows = []
    remaining_total_amt = float(amount_inr)

    for bucket in bucket_order:
        bucket_df = ranked[ranked["bucket"] == bucket].copy()
        if bucket_df.empty or remaining_total_amt <= 0:
            continue

        # Target capital for the bucket according to the profile weights.
        target_bucket_amt = float(amount_inr) * float(bucket_weights[bucket])
        bucket_remaining_amt = min(target_bucket_amt, remaining_total_amt)

        # Decide how many bonds to take from this bucket:
        # - at least 1 if there is any eligible bond
        # - cap by the band-derived "bonds per bucket"
        n_to_take = min(bonds_per_bucket, len(bucket_df))
        n_to_take = max(1, n_to_take)
        bucket_df = bucket_df.head(n_to_take).copy()

        # Allocate each chosen bond with whole lots.
        for idx, row in bucket_df.iterrows():
            if remaining_total_amt <= 0 or bucket_remaining_amt <= 0:
                break

            min_lot = float(row["min_lot_inr"])
            if min_lot <= 0:
                continue

            affordable_lots_by_bucket = int(math.floor(bucket_remaining_amt / min_lot))
            affordable_lots_by_total = int(math.floor(remaining_total_amt / min_lot))
            lots = min(affordable_lots_by_bucket, affordable_lots_by_total)

            # If the target bucket amount is below one lot but we still have total capital,
            # take a single lot to keep the portfolio operational.
            if lots <= 0 and remaining_total_amt >= min_lot:
                lots = 1

            if lots <= 0:
                continue

            allocation = lots * min_lot
            if allocation > remaining_total_amt:
                lots = int(math.floor(remaining_total_amt / min_lot))
                allocation = lots * min_lot
                if lots <= 0:
                    continue

            selected_rows.append(
                {
                    "bucket": bucket,
                    "isin": row["isin"],
                    "security_name": row["security_name"],
                    "credit_rating": row["credit_rating"],
                    "coupon_pct": row["coupon_pct"],
                    "ytm_pct": row["ytm_pct"],
                    "maturity_date": row["maturity_date"].date(),
                    "days_to_maturity": int(row["days_to_maturity"]),
                    "min_lot_inr": min_lot,
                    "qtm_inr": row["qtm_inr"],
                    "listed": row["listed"],
                    "duration_gap_days": int(row["duration_gap_days"]),
                    "yield_score": float(row["yield_score"]),
                    "liquidity_score": float(row["liquidity_score"]),
                    "issuer_score": float(row["issuer_score"]),
                    "platform_score": float(row["platform_score"]),
                    "bucket_weight": float(row["bucket_weight"]),
                    "allocation_inr": allocation,
                    "lots": lots,
                    "target_bucket_amt": target_bucket_amt,
                    "target_days": target_days,
                }
            )

            remaining_total_amt -= allocation
            bucket_remaining_amt -= allocation

    if not selected_rows:
        return pd.DataFrame(), pd.DataFrame()

    selected = pd.DataFrame(selected_rows)

    # If we selected multiple bonds in the same bucket, keep the sheet priority order.
    selected = selected.sort_values(
        by=["bucket", "duration_gap_days", "yield_score", "liquidity_score", "issuer_score"],
        ascending=[True, True, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Portfolio summary
    bucket_summary = (
        selected.groupby("bucket", as_index=False)
        .agg(
            allocation_inr=("allocation_inr", "sum"),
            lots=("lots", "sum"),
            ytm_pct=("ytm_pct", "mean"),
            avg_duration_gap_days=("duration_gap_days", "mean"),
        )
        .sort_values("bucket")
    )
    bucket_summary["pct"] = bucket_summary["allocation_inr"] / float(amount_inr) * 100.0

    return selected, bucket_summary


def format_inr(x: float) -> str:
    return f"₹{x:,.0f}"


def print_report(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    profile: str,
    amount_inr: float,
    duration_months: float,
) -> None:
    print("\n=== GOLDENPI RECOMMENDATIONS ===")
    print(f"Investment Amount : {format_inr(amount_inr)}")
    print(f"Risk Profile      : {profile}")
    print(f"Duration          : {duration_months:g} months")
    print(f"Raw Bonds Loaded  : {len(df)}")
    print(f"Eligible/Ranked   : {len(selected)}")

    if selected.empty:
        print("\nNo eligible bonds found after policy filters.")
        return

    weighted_ytm = (selected["allocation_inr"] * selected["ytm_pct"]).sum() / selected["allocation_inr"].sum()
    unallocated_cash = amount_inr - selected["allocation_inr"].sum()

    print("\nTop Ranked Bonds:")
    cols = [
        "isin",
        "security_name",
        "bucket",
        "credit_rating",
        "coupon_pct",
        "ytm_pct",
        "maturity_date",
        "days_to_maturity",
        "min_lot_inr",
        "qtm_inr",
        "duration_gap_days",
        "listed",
        "yield_score",
        "liquidity_score",
        "issuer_score",
        "bucket_weight",
    ]
    display_cols = [c for c in cols if c in selected.columns]
    print(selected[display_cols].head(10).to_string(index=False))

    print("\nSuggested Allocation by Bucket Logic:")
    alloc_cols = ["bucket", "security_name", "credit_rating", "allocation_inr", "lots", "min_lot_inr", "ytm_pct"]
    print(selected[alloc_cols].to_string(index=False))

    print(f"\nExpected Portfolio Weighted YTM: {weighted_ytm:.2f}%")
    print(f"Unallocated Cash: {format_inr(unallocated_cash)}")

    print("\nBucket Allocation Summary:")
    print(bucket_summary.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GoldenPi bond recommendation engine.")
    parser.add_argument("--file", required=True, help="Path to the GoldenPi Excel file.")
    parser.add_argument("--sheet", default=None, help="Sheet name. Defaults to first sheet.")
    parser.add_argument("--amount", type=float, required=True, help="Investment amount in INR.")
    parser.add_argument("--duration", type=float, required=True, help="Target duration in months.")
    parser.add_argument("--risk", default=None, help="Risk profile: Conservative, Moderate, Aggressive.")
    parser.add_argument("--risk-score", type=float, default=None, help="Optional numeric risk score.")
    parser.add_argument("--unlisted", choices=["yes", "no"], default="yes", help="Whether unlisted securities may be considered.")
    parser.add_argument("--top", type=int, default=5, help="Unused for selection cap; kept for CLI compatibility.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.risk_score is not None:
        profile = infer_profile_from_score(float(args.risk_score))
    elif args.risk:
        profile = args.risk.strip().title()
        if profile not in PROFILE_POLICIES:
            raise ValueError("Risk profile must be one of: Conservative, Moderate, Aggressive.")
    else:
        raise ValueError("Provide either --risk or --risk-score.")

    allow_unlisted = args.unlisted.lower() == "yes"

    df_raw = load_goldenpi(args.file, args.sheet)
    df = normalize_goldenpi(df_raw)

    ranked = ranked_candidates(
        df=df,
        profile=profile,
        amount_inr=float(args.amount),
        duration_months=float(args.duration),
        allow_unlisted=allow_unlisted,
    )

    selected, bucket_summary = choose_bonds(
        ranked=ranked,
        profile=profile,
        amount_inr=float(args.amount),
        duration_months=float(args.duration),
        allow_unlisted=allow_unlisted,
    )

    print_report(
        df=df,
        selected=selected,
        bucket_summary=bucket_summary,
        profile=profile,
        amount_inr=float(args.amount),
        duration_months=float(args.duration),
    )


if __name__ == "__main__":
    main()
