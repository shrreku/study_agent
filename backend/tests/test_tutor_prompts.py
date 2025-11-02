from __future__ import annotations

import json
from typing import Dict

import pytest

import os
import sys

# ensure root path for prompt/llm imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat


@pytest.fixture(scope="module")
def mock_context() -> Dict[str, str]:
    return {
        "concept": "Heat Transfer Fundamentals",
        "level": "beginner",
        "context": "[Chunk 1 | abc] Heat transfer describes energy flow due to temperature differences.",
        "student_message": "Can you explain conduction again?",
        "target_concepts": "Heat Transfer Fundamentals",
        "last_concept": "Conduction Basics",
    }


def _render_prompt(key: str, **vars):
    template = prompt_get(key)
    return prompt_render(template, vars)


def _call_prompt(prompt: str, default: Dict[str, object]) -> Dict[str, object]:
    resp = call_json_chat(prompt, default=default)
    assert isinstance(resp, dict)
    return resp


def test_tutor_classify_prompt_schema(mock_context):
    prompt = _render_prompt(
        "tutor.classify",
        student_message=mock_context["student_message"],
        target_concepts=mock_context["target_concepts"],
        last_concept=mock_context["last_concept"],
    )
    default = {
        "intent": "unknown",
        "affect": "neutral",
        "concept": mock_context["concept"],
        "confidence": 0.3,
        "needs_escalation": False,
    }
    result = _call_prompt(prompt, default)
    assert set(result.keys()) == {"intent", "affect", "concept", "confidence", "needs_escalation"}
    assert result["intent"] in {"question", "answer", "reflection", "off_topic", "greeting", "unknown"}
    assert result["affect"] in {"confused", "unsure", "engaged", "frustrated", "neutral"}
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_tutor_explain_prompt_schema(mock_context):
    prompt = _render_prompt(
        "tutor.explain",
        concept=mock_context["concept"],
        level=mock_context["level"],
        context=mock_context["context"],
    )
    default = {"response": "fallback", "confidence": 0.5}
    result = _call_prompt(prompt, default)
    assert set(result.keys()) == {"response", "confidence"}
    assert isinstance(result["response"], str)
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_tutor_ask_prompt_schema(mock_context):
    prompt = _render_prompt(
        "tutor.ask",
        concept=mock_context["concept"],
        level="developing",
        context=mock_context["context"],
    )
    default = {
        "question": "What is conduction?",
        "answer": "Heat transfer through solids",
        "confidence": 0.6,
        "options": [],
    }
    result = _call_prompt(prompt, default)
    assert set(result.keys()) == {"question", "answer", "confidence", "options"}
    assert isinstance(result["question"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["options"], list)
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_tutor_hint_prompt_schema(mock_context):
    prompt = _render_prompt(
        "tutor.hint",
        concept=mock_context["concept"],
        level="developing",
        context=mock_context["context"],
    )
    default = {"response": "Consider the temperature gradient.", "confidence": 0.5}
    result = _call_prompt(prompt, default)
    assert set(result.keys()) == {"response", "confidence"}
    assert isinstance(result["response"], str)
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_tutor_reflect_prompt_schema(mock_context):
    prompt = _render_prompt(
        "tutor.reflect",
        concept=mock_context["concept"],
        level="proficient",
        context=mock_context["context"],
    )
    default = {"response": "How would you teach this to a friend?", "confidence": 0.7}
    result = _call_prompt(prompt, default)
    assert set(result.keys()) == {"response", "confidence"}
    assert isinstance(result["response"], str)
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_tutor_prompts_default_render_is_valid_json(mock_context):
    keys = [
        (
            "tutor.classify",
            {
                "student_message": mock_context["student_message"],
                "target_concepts": mock_context["target_concepts"],
                "last_concept": mock_context["last_concept"],
            },
            json.dumps(
                {
                    "intent": "unknown",
                    "affect": "neutral",
                    "concept": mock_context["concept"],
                    "confidence": 0.3,
                    "needs_escalation": False,
                }
            ),
        ),
        (
            "tutor.explain",
            {
                "concept": mock_context["concept"],
                "level": "beginner",
                "context": mock_context["context"],
            },
            json.dumps({"response": "fallback", "confidence": 0.5}),
        ),
    ]
    for key, params, default_json in keys:
        prompt = _render_prompt(key, **params)
        assert prompt.strip()
        data = json.loads(default_json)
        assert isinstance(data, dict)
