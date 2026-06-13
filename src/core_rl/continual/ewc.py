from typing import Callable, Dict

import torch
import torch.nn as nn


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    Usage:
        # After task A is done:
        agent.consolidate(replay_buffer)

        # During task B training, update() automatically adds the penalty.
        # To tune protection strength:
        agent.ewc_lambda = 10000  # higher → stronger protection
    """

    def __init__(self, ewc_lambda: float = 5000.0) -> None:
        self.ewc_lambda = ewc_lambda
        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        self._consolidated = False

    def consolidate(
        self,
        model: nn.Module,
        states: torch.Tensor,
        log_prob_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Snapshot Fisher diagonal and optimal parameters for the current task.

        Args:
            model: Policy network to protect (e.g. SAC Actor).
            states: Representative state batch for this task.
            log_prob_fn: Callable(model, states) -> log_prob tensor shape (N, 1).
        """
        was_training = model.training
        model.eval()
        model.zero_grad()

        log_probs = log_prob_fn(model, states)
        (-log_probs.mean()).backward()

        self._fisher = {
            name: param.grad.detach().clone() ** 2
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        self._optimal_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self._consolidated = True

        model.zero_grad()
        model.train(was_training)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC regularization term to add to the actor loss.

        Returns 0 if consolidate() has not been called yet, so it is always
        safe to call — no if-guards needed at the call site.
        """
        if not self._consolidated:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                fisher = self._fisher[name].to(device)
                optimal = self._optimal_params[name].to(device)
                loss = loss + (fisher * (param - optimal).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * loss
