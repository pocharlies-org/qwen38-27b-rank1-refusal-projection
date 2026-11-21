#!/usr/bin/env python3
"""Run the paired probe module by module, committing each one as it lands.

WHY THIS EXISTS. `probe_rank1_candidates.py` writes its `--json` at the very end, so a run
that dies part way loses every number it already measured. That is what happened to both
pairs on the 19-11: the company pod restarted, the detached corrida went down with it, and
two logs of 0 bytes were all that was left. A pair is minutes of HTTP range reads, not
hours — but minutes is longer than the pod has been reliably staying up.

So this runner sequences the SAME measurement (it imports the probe module and calls
`probe_module`, `verdict`, `paired_rows`, `coherence`, `canon_sign`; it measures nothing
itself) and rewrites + commits the JSON after every module. A re-run reads the JSON back
and probes only what is missing, so a restart continues where it stopped instead of
starting over. A module that came out as an `error` row is retried, not skipped.

The u0 vectors go in a sidecar `.npz` next to the JSON — eight vectors of 2560 float64, a
few tens of KB — because `coherence` needs the directions of modules measured in an
EARLIER process, and the JSON deliberately carries no vectors.

    python3 tools/run_pair_modules.py --base REPO --base-rev <sha> --abl REPO --rev <sha> \\
        --module <key> [--module ...] --json results/DGX-324/results_parA.json \\
        --save-direction results/DGX-324/direction_parA.npy --against ref.npy
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_rank1_candidates as P  # noqa: E402 — the measurement lives there, not here


def git(wt: str, *args: str, retries: int = 3) -> bool:
    """Run git in the worktree, retrying a lock or a lost push a bounded number of times."""
    out = ""
    for attempt in range(retries):
        r = subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True)
        if r.returncode == 0:
            return True
        out = (r.stderr or r.stdout).strip()[:300]
        if args[0] == "commit" and "nothing to commit" in out:
            return True
        time.sleep(5 * (attempt + 1))
    print(f"  git {' '.join(args)} FAILED: {out}", flush=True)
    return False


def short(key: str) -> str:
    return (key.replace("model.language_model.layers.", "L")
               .replace("model.language_model.", "")
               .replace(".weight", ""))


def fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x))


def finish_candidate(st: dict, abl: str, mods: list[str], rows: dict, u0s: dict, args) -> None:
    """Recompute the aggregates over the rows measured so far, write them, commit them."""
    entry = st["candidates"][abl]
    ordered = [rows[k] for k in mods if k in rows]
    entry["modules"] = ordered
    entry["coherence"] = P.coherence([u0s[k] for k in mods if k in u0s]) or None
    entry["verdict"] = P.verdict(ordered)
    paired = P.paired_rows(ordered)
    entry["aggregate_paired"] = {
        "n_paired": len(paired),
        "min_s0_over_s1": min((r["s0_over_s1"] for r in paired), default=None),
        "max_s0_over_s1": max((r["s0_over_s1"] for r in paired), default=None),
        "n_format_mismatch": sum(1 for r in ordered if r.get("format_mismatch")),
    }
    entry["direction_module"] = args.direction_module
    entry["sign_rule"] = "largest-|component| entry positive, applied to both vectors"

    direction = None
    row = rows.get(args.direction_module) or {}
    if args.save_direction and u0s.get(args.direction_module) is not None and not row.get("format_mismatch"):
        u = u0s[args.direction_module]
        direction = P.canon_sign(u / np.linalg.norm(u))
        np.save(args.save_direction, direction.astype(np.float32))
    elif args.save_direction and pathlib.Path(args.save_direction).is_file():
        # Resumed run: the module was measured in an earlier process, its vector is on disk.
        direction = P.canon_sign(np.load(args.save_direction).astype(np.float64))

    cos_against = {}
    for ref in args.against or []:
        live = P.canon_sign(np.load(ref).astype(np.float64))
        if direction is None:
            cos_against[ref] = None
        elif live.size != direction.size:
            cos_against[ref] = None
        else:
            cos_against[ref] = float(live @ direction)
    entry["cos_direction_signed"] = cos_against

    json_path = pathlib.Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
    npz_path = json_path.with_name(json_path.stem + "_u0.npz")
    if u0s:
        np.savez(npz_path, **{k: u0s[k] for k in u0s})

    files = [str(json_path)]
    if args.save_direction and pathlib.Path(args.save_direction).is_file():
        files.append(args.save_direction)
    if npz_path.is_file():
        files.append(str(npz_path))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-rev", required=True, help="pin this when publishing a number")
    ap.add_argument("--abl", required=True)
    ap.add_argument("--rev", required=True, help="revision of --abl, pinned by SHA")
    ap.add_argument("--module", action="append", required=True)
    ap.add_argument("--json", required=True, help="results file, rewritten and committed per module")
    ap.add_argument("--save-direction", default=None, metavar="OUT.NPY")
    ap.add_argument("--direction-module", default="model.language_model.layers.0.linear_attn.out_proj.weight")
    ap.add_argument("--against", action="append", default=None, metavar="REF.NPY")
    ap.add_argument("--experts", default=None, metavar="0,1,7,255,511")
    ap.add_argument("--worktree", default=".", help="where to run git (the clone holding --json)")
    ap.add_argument("--branch", default=None, help="push target; commit only when absent")
    ap.add_argument("--label", default="pair", help="short name for the commit message")
    args = ap.parse_args()

    mods = args.module
    experts = tuple(int(x) for x in args.experts.split(",")) if args.experts else P.EXPERT_SAMPLE_DEFAULT
    abl, rev = args.abl, args.rev

    path = pathlib.Path(args.json)
    st = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    st["base"] = args.base
    st["base_rev"] = args.base_rev
    st["base_revision_sha"] = P.revision_sha(args.base, args.base_rev)
    st.setdefault("candidates", {})
    entry = st["candidates"].setdefault(abl, {})
    entry["rev"] = rev
    entry["revision_sha"] = P.revision_sha(abl, rev)
    rows = {r["key"]: r for r in entry.get("modules", []) if r.get("key")}
    npz_path = path.with_name(path.stem + "_u0.npz")
    u0s = {k: np.asarray(v, dtype=np.float64)
           for k, v in np.load(npz_path).items()} if npz_path.is_file() else {}

    print(f"base: {args.base}@{args.base_rev} -> {st['base_revision_sha']}")
    print(f"abl:  {abl}@{rev} -> {entry['revision_sha']}")
    print(f"modules: {len(mods)} ({len(rows)} already in {path.name})", flush=True)

    # Aggregates cover the whole pair, not just this invocation: a resumed run asked for
    # one module must not rewrite the verdict as if the other seven had never been read.
    all_mods = list(mods) + [k for k in rows if k not in set(mods)]

    for key in mods:
        if key in rows and not rows[key].get("error"):
            print(f"  {short(key):<34} already measured, skipped", flush=True)
            continue
        try:
            r = P.probe_module(args.base, abl, key, args.base_rev, rev, experts=experts)
        except Exception as e:  # noqa: BLE001 — one module must not take the pair down
            r = {"key": key, "error": f"{type(e).__name__}: {e}"}
        u = r.pop("_u0", None)
        r.pop("u0", None)
        if u is not None:
            u0s[key] = np.asarray(u, dtype=np.float64)
        rows[key] = r
        if r.get("error"):
            print(f"  {short(key):<34} ERROR: {r['error']}", flush=True)
        elif r.get("untouched"):
            print(f"  {short(key):<34} byte-identical to base (not edited)", flush=True)
        else:
            mm = " [FORMAT MISMATCH]" if r.get("format_mismatch") else ""
            print(f"  {short(key):<34} s0/s1={r['s0_over_s1']:8.2f}  rank1E={r['rank1E']:.4f}  "
                  f"cos={r['cos']:.4f}  lam_eff={r['lam_eff']:.4f}  delta={r['delta']:.5f}"
                  f"  floor={r['random_cos_floor']:.4f}{mm}", flush=True)
        files = finish_candidate(st, abl, all_mods, rows, u0s, args)
        git(args.worktree, "add", *files)
        git(args.worktree, "commit", "-m",
            f"DGX-1658: {args.label} — {short(key)} medido (revisiones fijadas por SHA)\n\n"
            "Co-Authored-By: Claude Code <noreply@anthropic.com>")
        if args.branch:
            git(args.worktree, "push", "origin", args.branch)

    finish_candidate(st, abl, all_mods, rows, u0s, args)
    entry = st["candidates"][abl]
    agg = entry["aggregate_paired"]
    print(f"\n{abl}: verdict={entry['verdict']}  paired={agg['n_paired']} "
          f"(mismatch excluded {agg['n_format_mismatch']})  "
          f"s0/s1 {fmt(agg['min_s0_over_s1'])} .. {fmt(agg['max_s0_over_s1'])}")
    for ref, v in (entry.get("cos_direction_signed") or {}).items():
        print(f"  cos_direction_signed vs {pathlib.Path(ref).name}: {fmt(v)}")
    print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
