"""
03 - CHAIN-OF-THOUGHT (CoT) PROMPTING
=====================================

Chain-of-thought = you ask the model to REASON STEP BY STEP before committing
to a final answer. Giving the model room to "think out loud" dramatically
improves accuracy on multi-step problems: math, logic, planning, debugging.

Two flavors:
  1. Zero-shot CoT  -> just add "Let's think step by step." / "Reason first."
  2. Few-shot CoT   -> show examples that include the reasoning, not just the
                       answer, so the model imitates the reasoning style.

KEY IDEAS
  - Reasoning helps most on problems with multiple dependent steps.
  - Ask for the FINAL ANSWER on its own line so it stays easy to parse.
  - Trade-off: more tokens + latency. For trivial tasks it's wasted cost.
  - For user-facing apps you often want the reasoning HIDDEN: have the model
    reason, then return only the conclusion (see the "answer only" variant).

Run:  python fundamentals/03_chain_of_thought.py
"""

# --- make the repo-root 'common' package importable when run directly ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import chat, header, rule

PROBLEM = (
    "A store had 120 apples. It sold 35% of them in the morning and 18 more in "
    "the afternoon. A delivery then added half of what was left. How many apples "
    "does the store have now?"
)


def no_cot() -> str:
    # Demanding an instant answer encourages a fast (often wrong) guess.
    return chat(
        [{"role": "user", "content": f"{PROBLEM}\nAnswer with just the number."}],
        temperature=0,
    )


def zero_shot_cot() -> str:
    prompt = (
        f"{PROBLEM}\n\n"
        "Work through this step by step, showing each calculation. "
        "Then on a final line write: ANSWER: <number>."
    )
    return chat([{"role": "user", "content": prompt}], temperature=0)


def reason_then_hide() -> str:
    """Reason internally, expose only the result -- common in production."""
    prompt = (
        f"{PROBLEM}\n\n"
        "Think through the steps silently. Do NOT show your work. "
        "Output only the final number."
    )
    return chat([{"role": "user", "content": prompt}], temperature=0)


if __name__ == "__main__":
    header("CHAIN-OF-THOUGHT PROMPTING")
    print("\nProblem:", PROBLEM)

    rule()
    print("\n[No reasoning, answer only] ->")
    print(no_cot())

    rule()
    print("\n[Zero-shot CoT: 'step by step'] ->")
    print(zero_shot_cot())

    rule()
    print("\n[Reason internally, return answer only] ->")
    print(reason_then_hide())

    rule()
    print(
        "\nTakeaway: letting the model reason before answering is one of the\n"
        "highest-leverage techniques for accuracy. Expose or hide the reasoning\n"
        "depending on whether your end user benefits from seeing it."
    )
