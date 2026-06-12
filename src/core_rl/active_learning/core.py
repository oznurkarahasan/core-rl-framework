import torch
from typing import Union


class ActiveLearningCore:
    """
    Agent-agnostic uncertainty detector.

    Receives a pre-computed entropy scalar and decides whether the current
    policy state is uncertain enough to warrant human review.
    Does not know or care whether the entropy came from SAC or PPO.

    Units: nats (natural log). Both agents expose entropy in nats:
      - SAC:  entropy ≈ -log_prob   (squashed Gaussian sample estimate)
      - PPO:  Categorical.entropy() (exact closed-form)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def is_uncertain(self, entropy: Union[float, torch.Tensor]) -> tuple[bool, float]:
        """
        Returns (is_uncertain, entropy_value).

        is_uncertain is True when entropy strictly exceeds the threshold,
        meaning the policy distribution is flat enough to warrant review.
        """
        entropy_val = entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy)
        return entropy_val > self.threshold, entropy_val
