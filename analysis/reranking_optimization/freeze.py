"""Choose and freeze one configuration using completed development results and budget."""

from . import evaluate
from .policies import POLICIES, policy_metadata
from .runner import FROZEN, OUTPUT, PARENT, TEMPLATE, now, read, remaining_seconds, sha, write


def budget_screen(manifest):
    projections = manifest["projections"]
    speed = manifest["measured_pairs_per_second"]
    remaining = remaining_seconds()
    if speed is None or speed <= 0:
        raise ValueError("Measured throughput is required")
    # A small margin allows for planning, artifacts, evaluation, and CPU variability.
    reserve_seconds = 180
    cost_margin = 1.05
    totals = {}
    for p in POLICIES:
        row = projections[p]
        heldout = row["remaining_heldout_pairs"]
        null_dev = row["remaining_null_dev_pairs"]
        null_heldout = row["remaining_null_heldout_pairs"]
        totals[p] = {
            "all_null": (heldout + null_dev + null_heldout) / speed,
            "dev_null": (heldout + null_dev) / speed,
            "heldout_only": heldout / speed,
        }
    for tier in ("all_null", "dev_null", "heldout_only"):
        eligible = [p for p in POLICIES if totals[p][tier] * cost_margin + reserve_seconds <= remaining]
        if eligible:
            return tier, eligible, totals, remaining, cost_margin, reserve_seconds
    return "heldout_budget_blocked", list(POLICIES), totals, remaining, cost_margin, reserve_seconds


def main():
    if FROZEN.exists() or (OUTPUT / "heldout_run_lock.json").exists():
        raise ValueError("A configuration has already been frozen")
    manifest = read(OUTPUT / "experiment_manifest.json")
    if not (OUTPUT / "dev_scores.jsonl").is_file() or not (OUTPUT / "development_performance.json").is_file():
        raise ValueError("Complete all 451 development queries before freezing")
    if read(OUTPUT / "development_performance.json")["queries"] != 451:
        raise ValueError("The selection cohort must contain all 451 development queries")
    tier, eligible, totals, left, margin, reserve = budget_screen(manifest)
    arguments = [
        "--stage", "development", "--checkpoint", str(OUTPUT / "dev_scores.jsonl"),
        "--throughput", str(manifest["measured_pairs_per_second"]), "--eligible-pools", *eligible,
    ]
    evaluate.main(arguments)
    decision_path = OUTPUT / "development_selection.json"
    decision = read(decision_path)
    winner = decision["recommendation"]
    if decision["query_count"] != 451 or winner["pool_id"] not in eligible:
        raise ValueError("Selection did not come from the full eligible development comparison")
    frozen = {
        "state": "frozen",
        "frozen_utc": now().isoformat(),
        "policy": winner["pool_id"],
        "scheme": winner["weight_id"],
        "weight_rrf": winner["w_rrf"],
        "weight_jina": 1,
        "rrf_k": 60,
        "template": TEMPLATE,
        "model": manifest["model"],
        "split_sha256": sha(PARENT / "splits.json"),
        "candidate_manifest_sha256": sha(PARENT / "candidate_pool_manifest.json"),
        "development_selection_sha256": sha(decision_path),
        "development_checkpoint_sha256": sha(OUTPUT / "dev_scores.jsonl"),
        "selection_cohort": "answerable_dev",
        "selection_query_count": 451,
        "budget_tier": tier,
        "eligible_pools": eligible,
        "budget_seconds_remaining_before_evaluation": left,
        "measured_pairs_per_second": manifest["measured_pairs_per_second"],
        "estimated_remaining_seconds": totals,
        "cost_margin": margin,
        "reserve_seconds": reserve,
        "pool_rank_policy": policy_metadata(),
    }
    write(FROZEN, frozen)
    print(f"Frozen {frozen['policy']}-{frozen['scheme']} for budget tier {tier}; eligible={eligible}", flush=True)


if __name__ == "__main__":
    main()
