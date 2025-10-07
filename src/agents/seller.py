"""
Seller as an agent. In this scenario, the seller bids a reserve price to maximize their utility.

Notation:
_________
other_bids: list of bids from buyer agents
cost: cost per item for the seller
"""

import numpy as np
from .base import BaseAgent, AuctionType


class SellerAgent(BaseAgent):
    def __init__(
        self,
        v: np.ndarray,
        k: int = 1,
        eps: float = 1e-3,
        eta: float = 1e-1,
        auction_type: AuctionType = AuctionType.SPA,
        alpha: float = 0,
        single_value: bool = True,
        cost: float = 0.0,
    ):
        super().__init__(v, k, 1.0, eps, eta, auction_type, alpha, single_value)
        self.cost = cost  # Cost of each item for the seller

    def allocate(self, other_bids: np.ndarray) -> np.ndarray:
        """Allocation function for the seller. Depending on the seller's reserve price bid, it determines how many items are sold.

        Note: We may change this to return an array with shape (N, k) where each column indicates whether the corresponding item is sold or not, if we want to implement item-specific reserve prices."""
        top_k_bids = other_bids[-self.k :]
        # Find the allocation corresponding to each possible bid action
        x = (self.bids[..., np.newaxis] <= top_k_bids).sum(axis=-1)
        return x  # (N,)

    def spa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """SPA payments function. The algebra below is a bit tricky... We first calculate the cumulative sums where the buyers pay price that comes before them."""

        top_k_bids = other_bids[-self.k :]
        top_2nd_k_bids = other_bids[-(self.k + 1) : -1]

        cumsum = np.insert(np.flip(top_2nd_k_bids[1:]).cumsum(), 0, 0)
        pass_count = (self.bids[..., np.newaxis] <= top_k_bids[:-1]).sum(axis=-1) - 1

        payment = cumsum[pass_count]
        # From the index of the cheapest second price, to the maximum bid, we should add
        s = np.searchsorted(self.bids, other_bids[-self.k], side="right")
        t = np.searchsorted(self.bids, other_bids[-1], side="right")
        payment[s:t] += self.bids[s:t]
        return payment

    def fpa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        other_bids.sort()
        top_k_bids = other_bids[-self.k :]

        cumsum = np.insert(np.flip(top_k_bids).cumsum(), 0, 0)
        pass_count = (self.bids[..., np.newaxis] <= top_k_bids).sum(axis=-1)
        return cumsum[pass_count]

    def objective(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Objective function. Returns the utility of each action."""
        u = p - self.cost * x
        return u

    def update_weights(self, other_bids: np.ndarray):
        print(f"Server Bids: {self.bids}")
        print(f"Server Bids.shape: {self.bids.shape}")

        other_bids.sort()
        other_bids = np.insert(other_bids, 0, 0)

        print(f"Buyer Bids: {other_bids}, len={len(other_bids)}")
        # Find the allocation and payments
        x = self.allocate(other_bids)

        print(f"x: {x}")
        print(f"x.shape: {x.shape}")

        p = self.payments(other_bids, x)
        print(f"p: {p}")
        print(f"p.shape: {p.shape}")

        # Calculate the objective function
        π = self.objective(x, p)

        # Update the weights
        self.weights *= 1 + self.eta * π
        self.weights /= np.sum(self.weights)

    def __repr__(self) -> str:
        return f"SellerAgent(v={self.v}, eps={self.eps}, eta={self.eta}, type={self.auction_type}, alpha={self.alpha}, cost={self.cost})"
