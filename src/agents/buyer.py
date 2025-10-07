"""
Notations:
_________
oponent_bids: list of bids from other buyer agents (i.e., excluding itself)
"""

from dataclasses import dataclass
import numpy as np
from .base import BaseAgent, BaseAgentConfig


@dataclass
class BuyerAgentConfig(BaseAgentConfig):
    tau: float = 1.0  # overbidding factor
    lam: float = 1  # hybrid objective function parameter
    single_value: bool = True


class BuyerAgent(BaseAgent):
    def __init__(self, config: BuyerAgentConfig):
        """Agent class that can bid and update its weights according to Multiplicative Weights (MW) Algorithm.

        Args:
            config: BuyerAgentConfig containing all agent parameters.
        """
        # Handle single_value logic before creating base config
        if len(config.v) == 1 and config.single_value and config.k > 1:
            v = np.array([config.v[0]] * config.k)
        else:
            v = config.v

        # Create base config with processed v
        base_config = BaseAgentConfig(
            v=v,
            k=config.k,
            eps=config.eps,
            eta=config.eta,
            auction_type=config.auction_type,
            alpha=config.alpha,
        )

        super().__init__(base_config)
        self.config = config
        self.tau = config.tau
        self.lam = config.lam

        # Recalculate bids with tau factor
        self.bids = np.arange(
            config.alpha,
            self.v.max() * config.tau + config.eps,
            config.eps,
            dtype=np.float32,
        )
        self.bids = self.bids.round(int(np.log10(1 / config.eps)))
        self.N = len(self.bids)
        self.weights = np.ones(self.N) / self.N

    def allocate(self, other_bids: np.ndarray) -> np.ndarray:
        """Allocation function. Returns the corresponding outcome, i.e., which item was won (or no item)"""
        top_k_bids = other_bids[-self.k :]

        # Find the allocation corresponding to each possible bid action
        x = np.digitize(self.bids, top_k_bids, right=True)
        # x = [0, 0, ..., 1, 1, ..., k, k] where 0 means no item won, 1 means the cheapest item won, and k means the most expensive item won
        return x  # (N,)

    def spa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """SPA payments function. Returns the corresponding payments for each action."""
        p = other_bids[-self.k :]  # Payments are top k bids
        p = np.insert(p, 0, 0)  # add a 0 to the beginning of the payments
        return p[x]  # (N,)

    def fpa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """FPA payments function. Returns the corresponding payments for each action."""
        # x != 0 means the agent won an item, so it pays its own bid
        p = self.bids.copy()
        p[x == 0] = 0
        return p  # (N,)

    def objective(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """(Hybrid) objective function. Th"""
        return self.v[x] - p * self.lam  # (N,)

    def update_weights(self, other_bids: np.ndarray):
        other_bids.sort()

        # Find the allocation and payments
        x = self.allocate(other_bids)
        p = self.payments(other_bids, x)

        # Calculate the objective function
        π = self.objective(x, p)

        # Update the weights
        self.weights *= 1 + self.eta * π
        self.weights /= np.sum(self.weights)

    def __repr__(self) -> str:
        return f"BuyerAgent(v={self.config.v}, eps={self.config.eps}, eta={self.config.eta}, type={self.config.auction_type}, alpha={self.config.alpha}, tau={self.config.tau}, lam={self.config.lam})"
