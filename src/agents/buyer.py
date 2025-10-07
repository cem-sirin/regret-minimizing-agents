"""
Notations:
_________
oponent_bids: list of bids from other buyer agents (i.e., excluding itself)
"""

import numpy as np
from .base import BaseAgent, AuctionType


class BuyerAgent(BaseAgent):
    def __init__(
        self,
        v: np.ndarray,
        k: int = 1,
        tau: float = 1.0,
        lam: float = 1,
        eps: float = 1e-3,
        eta: float = 1e-1,
        auction_type: AuctionType = AuctionType.SPA,
        alpha: float = 0,
        single_value: bool = True,
    ):
        """Agent class that can bid and update its weights according to Multiplicative Weights (MW) Algorithm.

        Args:
            v: value that the agent has for the item.
            k: number of items in the auction.
            tau: overbidding factor.
            lam: hybrid objective function parameter.
            eps: granularity of the bids.
            eta: learning rate of the MW algorithm.
            auction_type: type of the auction. Either "fpa" or "spa", i.e., first and second price auction.
            alpha: reserve price of the auction.
            single_value: whether the agent has a single value for multiple-item auctions.
        """
        super().__init__(v, k, tau, eps, eta, auction_type, alpha, single_value)
        self.lam = lam

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
        return f"Agent(v={self.v}, eps={self.eps}, eta={self.eta}, type={self.auction_type}, alpha={self.alpha})"
