import numpy as np
from typing import List, Union, Optional

"""
Notation:
_________
v: value of the item
k: number of items in the auction
lam: hybrid objective function parameter
eps: step size
eta: step size
alpha: reserve price (agents can't bid below this price)
N: number of actions (i.e., number of bid prices)
bids: corresponding bids for each action
weights: weights of each action
x: allocation
p: payments 

Notes:
______
Everything is sorted in ascending order. The values of items must be item_1 < item_2 < ... < item_k.
The allocation 0 means no items won. For example, if x = [0, 0, 1, 2], then bids 0 and 1 did not win 
any items, bid 2 won item 1, and bid 3 won item 2.

Auction types:
___
First Price Auction (fpa): winners pay their own bid
Second Price Auction (spa): winners pay the bid that comes after them
Uniform Price Auction (upa): winners pay the highest non-winning bid
Vickrey-Clarke-Groves Auction (vcga): winners pay the highest bid that is greater than their own bid
"""


class BaseAgent:
    def __init__(
        self,
        v: Union[float, List[float]],
        k: int = 1,
        tau: float = 1.0,
        eps: float = 1e-2,
        eta: float = 1e-2,
        type: str = "spa",
        alpha: float = 0,
        single_value: bool = True,
    ):
        """Base class for agents."""
        if isinstance(v, list):
            assert len(v) == k, "Length of v must be equal to k."
            v = np.array(v)
        else:
            # Agent has a single value for multiple-item auctions
            if single_value:
                v = np.array([v] * k)
            else:
                assert k == 1, "Agent was initialized with k > 1, but single_value is False and v is not a list."
                v = np.array([v])

        # v is a vector of k+1 size, where 0 is no items won. 1 is the first item won, 2 is the second item won, etc.
        v = np.insert(v, 0, 0)  # (k+1,)

        # Validation
        assert np.all(v[:-1] <= v[1:]), f"Input values must be ascending. v={v}"
        assert type in ["fpa", "spa"], "Type must be either 'fpa' or 'spa'."

        # Bidding agent configuration
        self.v = v
        self.k = k
        self.tau = tau
        self.eps = eps
        self.eta = eta
        self.alpha = alpha
        self.type = type
        self.single_value = single_value
        self.payments = getattr(self, type)

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


class BiddingAgent(BaseAgent):
    def __init__(
        self,
        v: Union[float, List[float]],
        k: int = 1,
        tau: float = 1.0,
        lam: float = 1,
        eps: float = 1e-3,
        eta: float = 1e-1,
        type: str = "spa",
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
            type: type of the auction. Either "fpa" or "spa", i.e., first and second price auction.
            alpha: reserve price of the auction.
            single_value: whether the agent has a single value for multiple-item auctions.
        """
        super().__init__(v, k, tau, eps, eta, type, alpha, single_value)
        self.lam = lam

    def allocate(self, oponent_bids: np.ndarray) -> np.ndarray:
        """Allocation function. Returns the corresponding outcome, i.e., which item was won (or no item)"""
        top_k_bids = oponent_bids[-self.k :]

        # Find the allocation corresponding to each possible bid action
        x = np.digitize(self.bids, top_k_bids, right=True)
        return x

    def spa(self, oponent_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        """SPA payments function. Returns the corresponding payments for each action."""
        p = oponent_bids[-self.k :]  # Payments are top k bids
        p = np.insert(p, 0, 0)  # add a 0 to the beginning of the payments
        return p[x]  # (N,)

    def fpa(self, _: np.ndarray, x: np.ndarray) -> np.ndarray:
        """FPA payments function. Returns the corresponding payments for each action."""
        p = self.bids.copy()
        p[x == 0] = 0
        return p  # (N,)

    def objective(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Hybrid objective function."""
        return self.v[x] - p * self.lam  # (N,)

    def update_weights(self, oponent_bids: float | List[float]):
        if isinstance(oponent_bids, list):
            oponent_bids = np.array(oponent_bids)
        else:
            oponent_bids = np.array([oponent_bids])

        oponent_bids.sort()

        # Find the allocation and payments
        x = self.allocate(oponent_bids)
        p = self.payments(oponent_bids, x)

        # print(f"b: {self.bids}")
        # print(f"x: {x.astype(float)}")
        # print(f"p: {p}")
        # print(f"v: {self.v[x]}")
        # print(f"π: {self.v[x] - p * self.lam}")

        # Calculate the objective function
        π = self.objective(x, p)

        # Update the weights
        self.weights *= 1 + self.eta * π
        self.weights /= np.sum(self.weights)

    def __repr__(self) -> str:
        return f"Agent(v={self.v}, eps={self.eps}, eta={self.eta}, type={self.type}, alpha={self.alpha})"


class ServerAgent(BaseAgent):
    def __init__(
        self,
        v: Union[float, List[float]],
        k: int = 1,
        eps: float = 1e-3,
        eta: float = 1e-1,
        type="spa",
        alpha: float = 0,
        single_value: bool = True,
    ):
        super().__init__(v, k, eps, eta, type, alpha, single_value)

    def allocate(self, buyer_bids: np.ndarray) -> np.ndarray:
        """Allocation function. Returns the corresponding outcome, i.e., which item was won (or no item)"""
        top_k_bids = buyer_bids[-self.k :]
        # Find the allocation corresponding to each possible bid action
        print(f"Top k bids: {top_k_bids}")
        x = np.digitize(self.bids, top_k_bids, right=True)
        return x

    def spa(self, buyer_bids: np.ndarray, x: np.ndarray) -> np.ndarray:
        # return np.maximum(second_bid, self.bids) * (self.bids <= highest_bid)
        top_k_bids = buyer_bids[-self.k :]
        top_k_bids = np.flip(top_k_bids)
        top_k_bids = np.insert(top_k_bids, 0, 0).cumsum()[::-1]

        print(f"Top k bids: {top_k_bids}")
        return top_k_bids[x]

    def fpa(self, highest_bid: float, second_bid: float) -> np.ndarray:
        return highest_bid * (self.bids <= highest_bid)

    def update_weights(self, buyer_bids: float | List[float]):

        print(f"Server Bids: {self.bids}")
        print(f"Server Bids.shape: {self.bids.shape}")

        if isinstance(buyer_bids, list):
            buyer_bids = np.array(buyer_bids)
        else:
            buyer_bids = np.array([buyer_bids])

        buyer_bids.sort()
        buyer_bids = np.insert(buyer_bids, 0, 0)

        print(f"Buyer Bids: {buyer_bids}, len={len(buyer_bids)}")
        # Find the allocation and payments
        x = self.allocate(buyer_bids)

        print(f"x: {x}")
        print(f"x.shape: {x.shape}")

        p = self.payments(buyer_bids, x)
        print(f"p: {p}")
        print(f"p.shape: {p.shape}")


class Auction:
    def __init__(
        self,
        type: str = "spa",
        n: Optional[int] = None,
        v_list: Optional[List[float]] = None,
        k: int = 1,
        alpha: float = 0,
        agent_args: Optional[dict] = {},
        include_server: bool = False,
    ):
        """Auction class.
        Args:
            type: type of the auction. Either "fpa" or "spa", i.e., first and second price auction.
            n: number of agents.
            k: number of items in the auction.
            alpha: reserve price of the auction.
            v_list: list of values for the agents. If not provided, the values will sampled uniformly from [0, 1].
            include_server: whether to include a server agent.
        """
        assert type in ["fpa", "spa"], "Type must be either 'fpa' or 'spa'."
        assert n >= 2, "Number of agents must be greater or equal to 2."

        assert n is not None or v_list is not None, "Either n or v_list must be provided."

        if v_list is None:
            v_list = np.random.uniform(0, 1, n)
        else:
            if len(v_list) != n:
                print(f"Warning: Length of v_list ({len(v_list)}) does not match n ({n}). Setting n to {len(v_list)}.")
            n = len(v_list)
            v_list = np.array(v_list)

        self.type = type
        self.n = n

        self.v_list = v_list
        self.bidders = [BiddingAgent(v, k=k, alpha=alpha, **agent_args) for v in v_list]

    def step(self):
        """A single iteration of the auction."""
        bids = {i: a.bid() for i, a in enumerate(self.bidders)}

        for i, a in enumerate(self.bidders):
            opponent_bids = [v for j, v in bids.items() if j != i]
            a.update_weights(opponent_bids)

        return bids

    def simulate(self, T: int = 5001):
        """Simulate the auction for T rounds."""

        history = []
        for _ in range(T):
            bids = self.step()
            history.append(bids)

        return history


if __name__ == "__main__":
    import pandas as pd
    import altair as alt

    np.set_printoptions(precision=1)

    auction = Auction(
        type="spa", n=3, v_list=[0.5, 0.6, 0.7, 1.0], k=2, alpha=0.1, agent_args={"eps": 1e-3, "tau": 2.0}
    )
    history = auction.simulate(T=1001)

    df = pd.DataFrame(history)
    df.columns = [f"Bidder {i+1} v={a.v.max()}" for i, a in enumerate(auction.bidders)]
    print(df.head())

    # Plot time series of bids using Altair
    df_long = df.melt(var_name="Bidder", value_name="Bid")
    df_long["Round"] = df_long.index
    chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(x="Round:Q", y="Bid:Q", color="Bidder:N")
        .properties(title="Bid Simulation")
    )
    chart.show()
