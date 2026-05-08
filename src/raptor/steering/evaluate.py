"""Evaluate steering generations with an LLM judge.

Input CSV columns expected from the steering generator:
``method, alpha, prompt, original_output, steered_output``.

The OpenAI API key is read from ``OPENAI_API_KEY``. The key is never stored in
this repository.
"""

import argparse
import csv
import os
import re
import time


USE_NEW_SDK = True
try:
    from openai import OpenAI

    client = OpenAI()
except Exception:
    import openai  # type: ignore

    USE_NEW_SDK = False


def ask_llm_judge(prompt: str, model: str) -> str:
    if USE_NEW_SDK:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def parse_scores(text: str):
    joy = coh = ""
    match = re.search(r"Joyfulness\s*[:=]\s*(\d)", text, re.IGNORECASE)
    if match:
        joy = match.group(1)
    match = re.search(r"Coherence\s*[:=]\s*(\d)", text, re.IGNORECASE)
    if match:
        coh = match.group(1)
    return joy, coh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="outputs/steering_results.csv")
    parser.add_argument("--out_csv", default="outputs/steering_evaluation.csv")
    parser.add_argument("--concept_desc", default="joyful")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    with open(args.in_csv, "r", encoding="utf-8") as fin, open(
        args.out_csv, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(
            [
                "method",
                "alpha",
                "prompt",
                "joyfulness_score",
                "coherence_score",
                "evaluation_comment",
            ]
        )

        for row in reader:
            method = row["method"]
            alpha = row["alpha"]
            prompt = row["prompt"]
            original = row["original_output"]
            steered = row["steered_output"]

            eval_prompt = (
                f"You are an impartial rater. A steered response should be more {args.concept_desc} than the original.\n"
                f"Prompt: {prompt}\n"
                f"Original response: {original}\n"
                f"Steered response: {steered}\n\n"
                "Rate the steered response for (1) Joyfulness and (2) Coherence, both from 1 to 5.\n"
                "Reply strictly in the format: Joyfulness=X, Coherence=Y, Comment=... (short)."
            )

            try:
                feedback = ask_llm_judge(eval_prompt, args.model)
            except Exception as exc:
                feedback = f"Error: {exc}"

            joy, coherence = parse_scores(feedback)
            writer.writerow([method, alpha, prompt, joy, coherence, feedback])
            print(
                f"[Eval] {method} alpha={alpha} -> Joyfulness={joy}, Coherence={coherence}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
