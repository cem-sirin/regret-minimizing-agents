import numpy as np
from typing import List, Optional

from ..agents.base import AuctionType
from ..agents.buyer import BuyerAgent


class Auction:
    def __init__(
        self,
        auction_type: AuctionType = AuctionType.SPA,
        n: Optional[int] = None,
        v_list: Optional[List[float] | np.ndarray] = None,
        k: int = 1,
        alpha: float = 0,
        agent_args: Optional[dict] = {},
        include_server: bool = False,
    ):
        """Auction class.
        Args:
            auction_type: type of the auction. Either "fpa" or "spa", i.e., first and second price auction.
            n: number of agents.
            v_list: list of values for the agents. If not provided, the values will sampled uniformly from [0, 1].
            k: number of items in the auction.
            alpha: reserve price of the auction.
            agent_args: arguments for the agents.
            include_server: whether to include a server agent.
        """
        assert n is not None or v_list is not None, (
            f"Either n or v_list must be provided. {n=}, {v_list=}"
        )

        if v_list is None:
            assert isinstance(n, int), "n must be an integer."
            v_list = np.random.uniform(0, 1, n)
        else:
            if len(v_list) != n:
                print(
                    f"Warning: Length of v_list ({len(v_list)}) does not match n ({n}). Setting n to {len(v_list)}."
                )
            v_list = np.array(v_list)

        if n is None:
            n = len(v_list)

        assert n >= 2, "Number of agents must be greater or equal to 2."

        self.auction_type = auction_type
        self.n = n
        self.alpha = alpha

        self.v_list = v_list
        self.bidders = [
            BuyerAgent(
                np.array([v]), k=k, auction_type=auction_type, alpha=alpha, **agent_args
            )
            for v in v_list
        ]

    def step(self):
        """A single iteration of the auction."""
        bids = {i: a.bid() for i, a in enumerate(self.bidders)}

        for i, a in enumerate(self.bidders):
            opponent_bids = [v for j, v in bids.items() if j != i]
            a.update_weights(np.array(opponent_bids))

        return bids

    def simulate(self, T: int = 5001):
        """Simulate the auction for T rounds."""

        history = []
        for _ in range(T):
            bids = self.step()
            history.append(bids)

        return history
