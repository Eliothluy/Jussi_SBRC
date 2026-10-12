#!/usr/bin/env python3
"""Stamp the score_version of crashed-eval sidecars so resumes stop re-running them.

Sidecars written by ``_failed_evaluation`` before the score_version fix carry the
default "v1" even in a v2 campaign. The resume cache rejects any sidecar whose
version differs from the requested one, so those evaluations were re-simulated on
every restart — violating the never-retry contract and, whenever the crash turned
out to be transient, silently diverging the optimizer trajectory from that
evaluation onward (invalidating every cached eval after it in that pair).

FAILED_SCORE is a constant hard penalty that no objective function computed, so
relabelling it is semantically correct: the value is identical under v1 and v2.
Only sidecars with ``failed: true`` are touched; successful evaluations carry a
genuinely version-dependent score and are never rewritten.

Re-run this after any campaign whose runners started before the fix landed, since
those processes keep writing "v1" on new failures.

    python3 examples/stamp_failed_sidecars.py --root <output_root> --score-version v2
    python3 examples/stamp_failed_sidecars.py --root <output_root> --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory to walk for candidate.json sidecars.")
    ap.add_argument("--score-version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = skipped = 0
    for sidecar in sorted(args.root.rglob("candidate.json")):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not payload.get("failed"):
            continue
        if payload.get("score_version") == args.score_version:
            skipped += 1
            continue
        payload["score_version"] = args.score_version
        rel = sidecar.relative_to(args.root)
        if args.dry_run:
            print(f"[dry-run] {rel}")
        else:
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            print(f"stamped   {rel}")
        changed += 1

    verb = "would stamp" if args.dry_run else "stamped"
    print(f"\n{verb} {changed} failed sidecar(s); {skipped} already at "
          f"score_version={args.score_version}")


if __name__ == "__main__":
    main()
