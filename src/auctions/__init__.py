import numpy as np
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from ..agents.base import AuctionType
from ..agents.buyer import BuyerAgent, BuyerAgentConfig
from ..agents.seller import SellerAgent, SellerAgentConfig


@dataclass
class AuctionConfig:
    auction_type: AuctionType = AuctionType.SPA
    n: Optional[int] = None
    v_list: Optional[List[float] | np.ndarray] = None
    k: int = 1
    alpha: float = 0
    agent_args: Optional[dict] = None
    include_seller: bool = False


class Auction:
    def __init__(self, config: AuctionConfig):
        """Auction class.
        Args:
            config: AuctionConfig containing all auction parameters.
        """
        self.config = config

        # Handle default agent_args
        agent_args = config.agent_args if config.agent_args is not None else {}

        assert config.n is not None or config.v_list is not None, (
            f"Either n or v_list must be provided. {config.n=}, {config.v_list=}"
        )

        if config.v_list is None:
            assert isinstance(config.n, int), "n must be an integer."
            v_list = np.random.uniform(0, 1, config.n)
        else:
            if len(config.v_list) != config.n:
                print(
                    f"Warning: Length of v_list ({len(config.v_list)}) does not match n ({config.n}). Setting n to {len(config.v_list)}."
                )
            v_list = np.array(config.v_list)

        if config.n is None:
            n = len(v_list)
        else:
            n = config.n

        assert n >= 2, "Number of agents must be greater or equal to 2."

        self.auction_type = config.auction_type
        self.n = n
        self.alpha = config.alpha

        self.v_list = v_list
        self.bidders = [
            BuyerAgent(
                BuyerAgentConfig(
                    v=np.array([v]),
                    k=config.k,
                    auction_type=config.auction_type,
                    alpha=config.alpha,
                    eps=agent_args.get("eps", 1e-3),
                    eta=agent_args.get("eta", 1e-1),
                    tau=agent_args.get("tau", 1.0),
                    lam=agent_args.get("lam", 1.0),
                    single_value=agent_args.get("single_value", True),
                )
            )
            for v in v_list
        ]

        # Add seller if include_seller is True
        self.seller = None
        if config.include_seller:
            self.seller = SellerAgent(
                SellerAgentConfig(
                    v=np.array([config.alpha]),  # Seller values are the reserve prices
                    k=config.k,
                    auction_type=config.auction_type,
                    alpha=config.alpha,
                    eps=agent_args.get("eps", 1e-3),
                    eta=agent_args.get("eta", 1e-1),
                    cost=agent_args.get("seller_cost", 0.0),
                )
            )

    def step(self) -> Dict[Any, float]:
        """A single iteration of the auction."""
        buyer_bids = {i: a.bid() for i, a in enumerate(self.bidders)}

        # Add seller bid if seller exists
        seller_bid = None
        if self.seller is not None:
            seller_bid = self.seller.bid()

        # Update buyer weights
        for i, a in enumerate(self.bidders):
            opponent_bids = [v for j, v in buyer_bids.items() if j != i]

            # If seller exists, add seller bid n times to opponent bids
            if self.seller is not None and seller_bid is not None:
                opponent_bids.extend([seller_bid] * self.n)

            a.update_weights(np.array(opponent_bids))

        # Update seller weights separately (only with buyer bids)
        if self.seller is not None:
            buyer_bids_list = [v for v in buyer_bids.values()]
            self.seller.update_weights(np.array(buyer_bids_list))

        # Return all bids using flexible typing
        all_bids: Dict[Any, float] = {}
        all_bids.update(buyer_bids)
        if self.seller is not None and seller_bid is not None:
            all_bids["seller"] = seller_bid

        return all_bids

    def simulate(self, T: int = 5001):
        """Simulate the auction for T rounds."""

        history = []
        for _ in range(T):
            bids = self.step()
            history.append(bids)

        return history
