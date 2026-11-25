"""Terminal interview: python -m openpath.interview.cli --mode build"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

from openpath.config import get_settings
from openpath.db import init_db
from openpath.interview.agent import Interview
from openpath.interview.llm import get_llm
from openpath.interview.store import save_profile, share_url
from openpath.profile import profile_summary


def run(interview: Interview, mode: str, need: str, input_fn: Callable = input, out: Callable = print,
        conn=None) -> dict:
    turn = interview.start(mode, need)
    while not turn.done:
        q = turn.question
        out(f"\n{q.text}")
        for i, opt in enumerate(q.options, 1):
            out(f"  {i}. {opt}")
        turn = interview.answer(turn.state, input_fn("> "))
    out("\nHere is what I understood (you can rerun to change anything):\n")
    out(profile_summary(turn.profile))
    if turn.used_fallback and interview.llm is not None:
        out("\n(The AI interviewer was unavailable, so the built-in questions were used.)")
    if conn is not None:
        out(f"\nShare code: {save_profile(conn, turn.profile)}")
    return turn.profile


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Describe what you need and answer a few questions.")
    p.add_argument("--mode", choices=["solve", "build", "learn"], required=True)
    p.add_argument("--need", help="what you need (asked interactively if omitted)")
    p.add_argument("--no-llm", action="store_true", help="use the built-in questions only")
    p.add_argument("--save", action="store_true", help="save the profile and print a share code")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    settings = get_settings()
    llm = None if args.no_llm else get_llm(settings)
    need = args.need or input("What do you need? ")
    conn = init_db(args.db or settings.db_path) if args.save else None
    run(Interview(llm), args.mode, need, conn=conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
