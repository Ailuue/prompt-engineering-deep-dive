"""
09 - SELF-CONSISTENCY (SAMPLING + VOTING)
=========================================

For problems with one correct answer but many reasoning paths, a single CoT run
can land on a wrong path. SELF-CONSISTENCY samples the SAME prompt several times
at a non-zero temperature, then takes a MAJORITY VOTE over the final answers.
The most common answer is usually the right one.

KEY IDEAS
  - Requires temperature > 0 so the samples actually differ.
  - Parse out the final answer from each sample, then vote.
  - Costs N x the tokens -> use it where correctness matters (math, extraction
    of a single fact, classification near a decision boundary).
  - It's a cheap, model-agnostic accuracy boost with no fine-tuning.

Run:  python fundamentals/09_self_consistency.py
"""

import re
from collections import Counter

# --- make the repo-root 'common' package importable when run directly ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import chat, header, rule

# A word problem people (and models) frequently slip on.
PROBLEM = (
    "I bought 3 boxes with 12 eggs each. I dropped one box and 5 eggs in another "
    "box were cracked. I used 7 good eggs to bake. How many good eggs are left?"
)

PROMPT = (
    f"{PROBLEM}\n\n"
    "Reason step by step, then end with a line exactly like: ANSWER: <number>"
)


def sample_answer() -> tuple[str, int | None]:
    text = chat([{"role": "user", "content": PROMPT}], temperature=0.8)
    match = re.search(r"ANSWER:\s*(-?\d+)", text)
    return text, (int(match.group(1)) if match else None)


def self_consistency(n: int = 5) -> None:
    answers: list[int] = []
    for i in range(n):
        _, ans = sample_answer()
        print(f"  sample {i + 1}: ANSWER = {ans}")
        if ans is not None:
            answers.append(ans)

    if not answers:
        print("No parseable answers.")
        return

    winner, count = Counter(answers).most_common(1)[0]
    print(f"\nMajority vote -> {winner}  ({count}/{len(answers)} samples agreed)")


if __name__ == "__main__":
    header("SELF-CONSISTENCY")
    print("\nProblem:", PROBLEM)

    rule()
    print("\nSingle sample (temperature 0.8) -- could be the unlucky wrong path:")
    text, ans = sample_answer()
    print(text)

    rule()
    print("\nSelf-consistency over 5 samples + majority vote:")
    self_consistency(5)

    rule()
    print(
        "\nTakeaway: when an answer is verifiable-by-agreement, sampling several\n"
        "reasoning paths and voting is a simple, reliable accuracy upgrade."
    )
