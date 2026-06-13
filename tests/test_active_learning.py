# tests/test_active_learning.py

import math
import pytest
import torch
import numpy as np
from core_rl.active_learning.core import ActiveLearningCore


# ---------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------

class TestActiveLearningCoreInit:
    def test_default_threshold(self):
        core = ActiveLearningCore()
        assert core.threshold == 0.5

    def test_custom_threshold(self):
        core = ActiveLearningCore(threshold=0.8)
        assert core.threshold == 0.8

    def test_threshold_is_mutable(self):
        core = ActiveLearningCore(threshold=0.5)
        core.threshold = 0.9
        assert core.threshold == 0.9


# ---------------------------------------------------------
# is_uncertain() — basic behaviour
# ---------------------------------------------------------

class TestIsUncertain:
    @pytest.fixture
    def core(self):
        return ActiveLearningCore(threshold=1.0)

    def test_returns_two_element_tuple(self, core):
        result = core.is_uncertain(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_uncertain_above_threshold(self, core):
        is_unc, val = core.is_uncertain(1.5)
        assert is_unc is True
        assert val == pytest.approx(1.5)

    def test_certain_below_threshold(self, core):
        is_unc, val = core.is_uncertain(0.3)
        assert is_unc is False
        assert val == pytest.approx(0.3)

    def test_equal_to_threshold_is_not_uncertain(self, core):
        # strictly greater than, so equality → certain
        is_unc, _ = core.is_uncertain(1.0)
        assert is_unc is False

    def test_entropy_value_returned_unchanged(self, core):
        _, val = core.is_uncertain(0.75)
        assert val == pytest.approx(0.75)

    def test_accepts_float_input(self, core):
        is_unc, val = core.is_uncertain(0.5)
        assert isinstance(val, float)

    def test_accepts_tensor_input(self, core):
        entropy_tensor = torch.tensor(1.5)
        is_unc, val = core.is_uncertain(entropy_tensor)
        assert is_unc is True
        assert isinstance(val, float)

    def test_tensor_value_matches_float(self, core):
        entropy_tensor = torch.tensor(0.42)
        _, val = core.is_uncertain(entropy_tensor)
        assert val == pytest.approx(0.42, abs=1e-5)


# ---------------------------------------------------------
# Threshold mutability
# ---------------------------------------------------------

class TestThresholdMutability:
    def test_lowering_threshold_flips_decision(self):
        core = ActiveLearningCore(threshold=1.0)
        is_unc_before, _ = core.is_uncertain(0.8)
        core.threshold = 0.5
        is_unc_after, _ = core.is_uncertain(0.8)
        assert is_unc_before is False
        assert is_unc_after is True

    def test_raising_threshold_flips_decision(self):
        core = ActiveLearningCore(threshold=0.3)
        is_unc_before, _ = core.is_uncertain(0.8)
        core.threshold = 1.0
        is_unc_after, _ = core.is_uncertain(0.8)
        assert is_unc_before is True
        assert is_unc_after is False


# ---------------------------------------------------------
# Integration: realistic entropy ranges from agents
# ---------------------------------------------------------

class TestEntropyRanges:
    """
    Categorical distribution over n actions has:
      min entropy = 0  (deterministic, one action = 1.0)
      max entropy = ln(n) (uniform)

    SAC Gaussian entropy is unbounded above but typically positive
    for well-initialised networks.
    """

    def test_ppo_uniform_policy_is_uncertain(self):
        # ln(4) ≈ 1.386 — maximally uncertain Categorical over 4 actions
        core = ActiveLearningCore(threshold=0.5)
        max_entropy = math.log(4)
        is_unc, val = core.is_uncertain(max_entropy)
        assert is_unc is True
        assert val == pytest.approx(max_entropy, rel=1e-4)

    def test_ppo_deterministic_policy_is_certain(self):
        core = ActiveLearningCore(threshold=0.5)
        is_unc, _ = core.is_uncertain(0.01)
        assert is_unc is False

    def test_sac_high_entropy_detected(self):
        # SAC alpha starts near 1.0; -log_prob can be > 2 for spread Gaussian
        core = ActiveLearningCore(threshold=1.5)
        is_unc, _ = core.is_uncertain(2.0)
        assert is_unc is True

    def test_zero_entropy_always_certain(self):
        core = ActiveLearningCore(threshold=0.0)
        # Threshold=0.0: only entropy > 0 triggers uncertainty
        is_unc, _ = core.is_uncertain(0.0)
        assert is_unc is False
