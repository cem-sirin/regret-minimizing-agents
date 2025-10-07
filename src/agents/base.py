from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class AuctionType(Enum):
    FPA = "fpa"
    SPA = "spa"


@dataclass
class BaseAgentConfig:
    v: np.ndarray
    k: int = 1
    eps: float = 1e-2
    eta: float = 1e-2
    auction_type: AuctionType = AuctionType.SPA
    alpha: float = 0


class BaseAgent(ABC):
    def __init__(self, config: BaseAgentConfig):
        """Base class for agents."""
        self.config = config

        assert len(config.v.shape) == 1, "v must be a 1D array."
        assert len(config.v) == config.k, "Length of v must be equal to k."

        # v is a vector of k+1 size, where 0 is no items won. 1 is the first item won, 2 is the second item won, etc.
        self.v = np.insert(config.v, 0, 0)  # (k+1,)

        # Validation
        assert np.all(self.v[:-1] <= self.v[1:]), (
            f"Input values must be ascending. v={self.v}"
        )

        # Bidding agent configuration
        self.k = config.k
        self.eps = config.eps
        self.eta = config.eta
        self.alpha = config.alpha
        self.auction_type = config.auction_type
        self.payments = getattr(self, config.auction_type.value)

        # N: number of actions (i.e., number of bid prices)
        # bids: corresponding bids for each action
        # weights: weights of each action
        self.bids = np.arange(
            config.alpha, 1.0 + config.eps, config.eps, dtype=np.float32
        )
        self.bids = self.bids.round(int(np.log10(1 / config.eps)))
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
