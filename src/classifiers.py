from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import requests


LABELS = [
    "account",
    "billing",
    "delivery",
    "product_usage",
    "technical_issue",
    "refund",
]


class Classifier(Protocol):
    def predict(self, query: str) -> str:
        ...


@dataclass
class RuleBasedClassifier:
    """A deliberately simple baseline for reproducible offline evaluation."""

    def predict(self, query: str) -> str:
        text = query.lower()

        rules: list[tuple[str, Sequence[str]]] = [
            ("refund", ["refund", "money back", "return", "reimburs", "payment reversed"]),
            ("delivery", ["shipping", "shipment", "courier", "tracking", "parcel", "delivered", "pickup point"]),
            ("billing", ["charged", "invoice", "payment", "subscription", "renewal", "coupon", "tax", "card"]),
            ("technical_issue", ["crash", "error", "502", "freeze", "blank screen", "timeout", "timing out", "failed", "loading forever"]),
            ("account", ["account", "password", "sign in", "login", "profile", "verification code", "two-factor", "phone number"]),
            ("product_usage", ["export", "workspace", "invite", "dashboard", "schedule", "api settings", "duplicate", "notification"]),
        ]

        for label, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return label
        return "product_usage"


class OpenAICompatibleClassifier:
    """Optional classifier for any OpenAI-compatible chat-completions endpoint.

    Required environment variables:
      - LLM_BASE_URL, e.g. https://api.example.com/v1
      - LLM_API_KEY
      - LLM_MODEL
    """

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.timeout_seconds = timeout_seconds
        missing = [
            name
            for name, value in {
                "LLM_BASE_URL": self.base_url,
                "LLM_API_KEY": self.api_key,
                "LLM_MODEL": self.model,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    def predict(self, query: str) -> str:
        system_prompt = (
            "You are an intent classifier. Return only JSON in the form "
            '{"label":"<label>"}. Valid labels: ' + ", ".join(LABELS) + "."
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        label = self._parse_label(content)
        if label not in LABELS:
            raise ValueError(f"Unexpected label from model: {label!r}")
        return label

    @staticmethod
    def _parse_label(content: str) -> str:
        try:
            return json.loads(content)["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            match = re.search(r'"?label"?\s*[:=]\s*"?([a-z_]+)', content.lower())
            if not match:
                raise ValueError(f"Could not parse model output: {content!r}")
            return match.group(1)
