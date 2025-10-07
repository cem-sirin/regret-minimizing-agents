from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class AuctionType(Enum):
    FPA = "fpa"
    SPA = "spa"


class BaseAgent(ABC):
    def __init__(
        self,
        v: np.ndarray,
        k: int = 1,
        tau: float = 1.0,
        eps: float = 1e-2,
        eta: float = 1e-2,
        auction_type: AuctionType = AuctionType.SPA,
        alpha: float = 0,
        single_value: bool = True,
    ):
        """Base class for agents."""
        assert len(v.shape) == 1, "v must be a 1D array."

        if len(v) == 1 and single_value and k > 1:
            v = np.array([v[0]] * k)

        assert len(v) == k, "Length of v must be equal to k."

        # v is a vector of k+1 size, where 0 is no items won. 1 is the first item won, 2 is the second item won, etc.
        v = np.insert(v, 0, 0)  # (k+1,)

        # Validation
        assert np.all(v[:-1] <= v[1:]), f"Input values must be ascending. v={v}"

        # Bidding agent configuration
        self.v = v
        self.k = k
        self.tau = tau
        self.eps = eps
        self.eta = eta
        self.alpha = alpha
        self.auction_type = auction_type
        self.single_value = single_value
        self.payments = getattr(self, auction_type.value)

        # N: number of actions (i.e., number of bid prices)
        # bids: corresponding bids for each action
        # weights: weights of each action
        self.bids = np.arange(alpha, v.max() * tau + eps, eps, dtype=np.float32)
        self.bids = self.bids.round(int(np.log10(1 / eps)))
        self.N = len(self.bids)
        self.weights = np.ones(self.N) / self.N

    def choose_action(self) -> int:
        """Choose action with probability proportional to weights."""
        return np.random.choice(self.N, p=self.weights / np.sum(self.weights))

    def bid(self) -> float:
        """Return the bid of the chosen action."""
        return self.bids[self.choose_action()]

    @abstractmethod
    def allocate(self, other_bids: np.ndarray) -> np.ndarray:
        """Allocation function. Returns the corresponding outcome, i.e., which item was won (or no item)"""
        pass

    @abstractmethod
    def spa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """SPA payments function. Returns the corresponding payments for each action."""
        pass

    @abstractmethod
    def fpa(self, other_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """FPA payments function. Returns the corresponding payments for each action."""
        pass

    @abstractmethod
    def objective(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Objective function."""
        pass

    @abstractmethod
    def update_weights(self, other_bids: np.ndarray):
        """Update weights according to the Multiplicative Weights (MW) Algorithm."""
        pass
