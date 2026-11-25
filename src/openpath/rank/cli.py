"""Get recommendations: python -m openpath.rank.cli --code abcd2345   (or --mode build --need "...")"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Optional

from openpath.config import get_settings
from openpath.db import init_db
from openpath.index.embedder import get_embedder
from openpath.index.search import Searcher
from openpath.interview.llm import get_llm
from openpath.interview.store import load_profile, save_profile
from openpath.profile import profile_summary
from openpath.rank.config import load_rank_config
from openpath.rank.ranker import RankResult, rank, save_run


def format_result(i: int, r: Any) -> str:
    p, h, c, e = r.project, r.health, r.confidence, r.explanation
    score = "n/a" if h.score_total is None else f"{h.score_total:.0f}"
    lines = [f"\n{i}. {p['name']}  [{p['kind']}/{p['source']}]  match score {r.score:.0f}/100",
             f"   Health {score} ({h.label}) | License {r.license.summary} | Effort: {e.effort['level']} | "
             f"Confidence: {c.label} - {c.message}",
             "   Why it fits:"]
    lines += [f"     - {s}" for s in e.fit]
    if e.tradeoffs:
        lines.append("   Watch out for:")
        lines += [f"     - {s}" for s in e.tradeoffs]
    if e.effort.get("note"):
        lines.append(f"   Effort note: {e.effort['note']}")
    lines.append(f"   {p['url']}")
    return "\n".join(lines)


def format_result_set(profile: dict, res: RankResult) -> str:
    m = res.meta
    out = [f"Recommendations for: {profile['need']}",
           f"({m['pool']} close candidates checked, {m['excluded_by_rules']} excluded by your requirements, "
           f"{m['verified']} verified live)"]
    out += [f"Note: {w}" for w in m["warnings"]]
    out += [format_result(i, r) for i, r in enumerate(res.results, 1)]
    if res.rejected:
        out.append("\nWhy not these? (close matches that were excluded)")
        for r in res.rejected:
            out.append(f"  - {r.project['name']} ({r.project['source_id']}): " + " ".join(x.text for x in r.reasons))
    return "\n".join(out)


def main(argv: Optional[list] = None, out: Callable = print) -> int:
    p = argparse.ArgumentParser(description="Recommend open source projects for a requirement profile.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--code", help="share code from the interview")
    src.add_argument("--profile-file", help="JSON file with a requirement profile")
    src.add_argument("--need", help="quick search without constraints (use with --mode)")
    p.add_argument("--mode", choices=["solve", "build", "learn"], default="build")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--no-verify", action="store_true", help="skip live checks (use saved data only)")
    p.add_argument("--no-llm", action="store_true", help="template explanations only")
    p.add_argument("--embedder", default=None)
    p.add_argument("--save-run", action="store_true")
    p.add_argument("--config", help="custom ranking config JSON")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    settings = get_settings()
    conn = init_db(args.db or settings.db_path)
    if args.code:
        profile = load_profile(conn, args.code)
        if profile is None:
            print(f"No profile found for code {args.code!r}", file=sys.stderr)
            return 1
    elif args.profile_file:
        with open(args.profile_file, encoding="utf-8") as fh:
            profile = json.load(fh)
    else:
        profile = {"mode": args.mode, "need": args.need, "constraints": {}, "preferences": {}}

    verifier = None
    if not args.no_verify:
        from huggingface_hub import HfApi

        from openpath.ingest.github import GitHubClient
        from openpath.rank.verify import LiveVerifier
        verifier = LiveVerifier(conn, GitHubClient(settings.github_token, conn), HfApi(token=settings.hf_token or None))
    searcher = Searcher(conn, get_embedder(args.embedder or settings.embedder))
    res = rank(conn, profile, searcher, verifier=verifier, llm=None if args.no_llm else get_llm(settings),
               cfg=load_rank_config(args.config), k=args.k)
    if args.json:
        out(json.dumps(res.to_dict(), indent=1))
    else:
        out(profile_summary(profile) + "\n")
        out(format_result_set(profile, res))
    if args.save_run:
        code = save_profile(conn, profile)
        out(f"\nSaved run {save_run(conn, code, res)} for profile code {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
