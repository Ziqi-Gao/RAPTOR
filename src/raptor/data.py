from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset

from raptor.model_utils import add_noise


class DataProcessing:
    """Load the six binary concept datasets used in the RAPTOR benchmark."""

    def __init__(self, data_path: str, data_name: str, noise: str = "non-noise"):
        self.data_path = Path(data_path) if data_path else Path()
        self.data_name = data_name
        self.noise = noise

    def dispacher(self) -> Tuple[list[str], list[str], str, Optional[str]]:
        # Keep the misspelled method name for compatibility with older scripts.
        return self.dispatcher()

    def dispatcher(self) -> Tuple[list[str], list[str], str, Optional[str]]:
        loaders = {
            "STSA": (self.stsa, "The sentence above is a movie review. Judge whether its sentiment is Positive or Negative."),
            "sarcasm": (self.sarcasm, "Detect whether this sentence is sarcastic. Answer Yes or No."),
            "hatexplain": (self.hatexplain, "Tell whether the comment presents hate speech or offensive content."),
            "counterfact": (self.counterfact, "Judge whether the statement is true or false."),
            "cities": (self.cities, "Judge whether the statement is true or false."),
            "common": (self.common, "Judge whether the statement is true or false."),
        }
        if self.data_name not in loaders:
            raise ValueError(f"Unknown RAPTOR dataset: {self.data_name}")
        loader, prompt = loaders[self.data_name]
        pos, neg = loader()
        print(
            f"### Dataset: {self.data_name} | path: {self.data_path} | "
            f"positive={len(pos)} negative={len(neg)} ###"
        )
        return pos, neg, prompt, None

    def get_prompt(self, prompt: str, cot: Optional[str], question: str) -> str:
        del cot
        if self.data_name in {"common", "cities", "counterfact"}:
            return f"{prompt} {question}"
        return f"{question} {prompt}"

    def _maybe_noise(self, text: str) -> str:
        return add_noise(text) if self.noise == "noise" else text

    def stsa(self) -> Tuple[list[str], list[str]]:
        pos, neg = [], []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_raw, text = line.split(" ", 1)
                (pos if int(label_raw) == 1 else neg).append(self._maybe_noise(text))
        return pos, neg

    def sarcasm(self) -> Tuple[list[str], list[str]]:
        pos, neg = [], []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = self._maybe_noise(row["headline"])
                (pos if int(row["is_sarcastic"]) == 1 else neg).append(text)
        return pos[:3000], neg[:3000]

    def hatexplain(self) -> Tuple[list[str], list[str]]:
        pos_labels = {0, 2}  # hatespeech and offensive are positive.
        label_map = {"hatespeech": 0, "normal": 1, "offensive": 2}

        if self.data_path and self.data_path.is_dir():
            ds = load_dataset(str(self.data_path), split="train")
        else:
            ds = load_dataset("hatexplain", split="train")

        pos, neg = [], []
        for row in ds:
            text = self._maybe_noise(" ".join(row.get("post_tokens", [])))
            annotators = row.get("annotators", [])
            if isinstance(annotators, dict):
                annotators = annotators.get("label", [])

            labels = []
            for ann in annotators:
                lab = ann.get("label") if isinstance(ann, dict) else ann
                if isinstance(lab, str):
                    lab = label_map.get(lab, lab)
                try:
                    labels.append(int(lab))
                except (TypeError, ValueError):
                    continue
            if not labels:
                continue

            counts = [0, 0, 0]
            for lab in labels:
                if 0 <= lab < len(counts):
                    counts[lab] += 1
            max_count = max(counts)
            if counts.count(max_count) > 1:
                continue
            (pos if counts.index(max_count) in pos_labels else neg).append(text)
        return pos, neg

    def counterfact(self) -> Tuple[list[str], list[str]]:
        df = pd.read_csv(self.data_path)
        pos = [self._maybe_noise(x) for x in df[df["label"] == 1]["statement"].tolist()]
        neg = [self._maybe_noise(x) for x in df[df["label"] == 0]["statement"].tolist()]
        return pos[:2000], neg[:2000]

    def cities(self) -> Tuple[list[str], list[str]]:
        df = pd.read_csv(self.data_path)
        pos = [self._maybe_noise(x) for x in df[df["label"] == 1]["statement"].tolist()]
        neg = [self._maybe_noise(x) for x in df[df["label"] == 0]["statement"].tolist()]
        return pos, neg

    def common(self) -> Tuple[list[str], list[str]]:
        df = pd.read_csv(self.data_path)
        pos = [self._maybe_noise(x) for x in df[df["label"] == "True"]["examples"].tolist()]
        neg = [self._maybe_noise(x) for x in df[df["label"] == "False"]["examples"].tolist()]
        return pos[:3000], neg[:3000]
