from __future__ import annotations

import os
import sys

# ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.tools.mastery_updater import MasteryUpdater


def test_mastery_increase_on_correct_answer_basic():
    updater = MasteryUpdater()
    signals = {"affect": "engaged", "intent": "answer", "answer_correct": True}
    update = updater.compute_mastery_delta(
        concept="Conduction",
        user_id="00000000-0000-0000-0000-000000000001",
        interaction_signals=signals,
        current_mastery=0.5,
    )
    assert update.delta >= 0.0
    assert "correct_answer" in update.reason or update.confidence >= 0.5


def test_mastery_decrease_or_zero_on_confusion():
    updater = MasteryUpdater()
    signals = {"affect": "confused", "intent": "question"}
    update = updater.compute_mastery_delta(
        concept="Fourier Law",
        user_id="00000000-0000-0000-0000-000000000001",
        interaction_signals=signals,
        current_mastery=0.6,
    )
    # With default learning_rate/min_update, delta may clamp to 0.0; allow non-positive
    assert update.delta <= 0.0
