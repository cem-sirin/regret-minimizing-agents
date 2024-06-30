import streamlit as st
import numpy as np
from typing import List
import pandas as pd


# streamlit run app.py
class Agent:
    def __init__(
        self,
        v: float,
        eps: float = 1e-3,
        eta: float = 1e-1,
        type="spa",
        alpha: float = 0,
    ):
        """Agent class that can bid and update its weights according to Multiplicative Weights (MW) Algorithm.

        Args:
            v: value that the agent has for the item.
            eps: granularity of the bids.
            eta: learning rate of the MW algorithm.
            type: type of the MW algorithm. Either "fpa" or "spa", i.e., first and second price auction.
            alpha: minimum bid value.
        """
        self.v = v
        self.eps = eps
        self.eta = eta

        assert type in ["fpa", "spa"], "Type must be either 'fpa' or 'spa'."
        self.type = type

        # N: number of actions
        # bids: corresponding bids for each action
        # weights: weights of each action
        self.N = round((v - alpha) / eps + 1)
        self.bids = np.arange(alpha, v + eps, eps).round(int(np.log10(1 / eps)))
        # TODO: Integerize the bids
        self.weights = np.ones(self.N)

    def choose_action(self) -> int:
        """Choose action with probability proportional to weights."""
        return np.random.choice(self.N, p=self.weights / np.sum(self.weights))

    def bid(self) -> float:
        """Return the bid of the chosen action."""
        return self.bids[self.choose_action()]

    def _min_selector(self, u: np.ndarray) -> np.ndarray:
        u[u == 0] = np.inf
        u = np.min(u, axis=1)
        u[u == np.inf] = 0
        return u

    def spa(self, oponent_bids: np.ndarray) -> np.ndarray:
        u = (self.v - oponent_bids) * (self.bids[:, None] > oponent_bids)
        return self._min_selector(u)

    def fpa(self, oponent_bids: np.ndarray) -> np.ndarray:
        u = (self.v - self.bids[:, None]) * (self.bids[:, None] > oponent_bids)
        return self._min_selector(u)

    def update_weights(self, oponent_bids: float | np.ndarray):
        if isinstance(oponent_bids, (int, float)):
            oponent_bids = np.array([oponent_bids])

        # Calculate utility of each action if it had been played
        u = getattr(self, self.type)(oponent_bids)
        # Update weights
        self.weights *= 1 + self.eta * u
        self.weights /= np.sum(self.weights)


def simulate_generalized_auction(agents: List[Agent], k: int = 2, T: int = 5001):
    assert len(agents) > k, "Number of agents must be greater than number of items."
    history = []
    for _ in range(T):
        bids = [(a.bid(), i) for i, a in enumerate(agents)]
        history.append([b[0] for b in bids])
        bids = sorted(bids, reverse=True)

        for _, i in bids[:k]:  # top k bidding agents (winners)
            # Put k-1 highest bids in an array except the current agent
            highest_bids = np.array([b for b, j in bids[: k + 1] if j != i])
            assert len(highest_bids) == k, "Number of highest bids must be k."
            agents[i].update_weights(highest_bids)

        for b, i in bids[k:]:  # losing agents
            # Put k-1 highest bids in an array
            highest_bids = np.array([b for b, j in bids][:k])
            assert len(highest_bids) == k, "Number of highest bids must be k."
            agents[i].update_weights(highest_bids)

    return history


def simulate_vickrey_auction(agents: List[Agent], k: int = 2, T=5001):
    assert len(agents) > k, "Number of agents must be greater than number of items."
    history = []
    for _ in range(T):
        bids = [(a.bid(), i) for i, a in enumerate(agents)]
        history.append([b[0] for b in bids])
        bids = sorted(bids, reverse=True)

        # The tipping point is the (k+1)th highest bid for top k agents
        for a in agents[:k]:
            a.update_weights(bids[k][0])

        # and the tipping point is the kth highest bid for top k agents
        for a in agents[k:]:
            a.update_weights(bids[k - 1][0])

    return history


st.title("Regret Minimizing Agents")
st.write(
    "This is a simple web app to simulate regret minimizing agents in auctions. Agents minimize their regret by updating their weights using the Multiplicative Weights (MW) Algorithm."
    + " The weights are updated simply by $$w_{i,t+1} = w_{i,t} (1 + \eta u_{i,t})$$ where $u_{i,t}$ is the potential utility of action $i$ if it had been played at time $t$ and $\eta$ is the learning rate. At each round, agents randomly choose an action with probability proportional to their weights and update their weights based on the outcome of the auction."
)

st.write("### Simulation Parameters")

# Auction type
auction_type = st.radio("Auction Type", ["First Price Auction", "Second Price Auction", "Vickrey Auction"])

# Add button for adjusting the number of agents
# num_agents = st.slider("Number of Agents", 1, 5, 3)

# Replace a text box for agent inputs to adjust values
vals = st.text_input("Values of Agents", "1.0, 0.6, 0.3")
num_agents = len(vals.split(","))
vals = [float(v) for v in vals.split(",")]


# Add button to adjust number of items
k = st.number_input("Number of Items (max: num_agents - 1)", 1, num_agents - 1, 2)

# Add button for adjusting the number of rounds
T = st.slider("Number of Rounds", 1, 5000, 500)
# Reserve price
alpha = st.number_input("Reserve Price", 0.0, 1.0, 0.20, 0.01)

# Add button to start the simulation
if st.button("Start Simulation"):
    # agents = [Agent(np.random.rand()) for _ in range(num_agents)]
    if k >= num_agents:
        st.error("Number of items must be less than the number of agents. Please adjust the number of items.")
    else:

        agents = []
        type = "fpa" if auction_type == "First Price Auction" else "spa"
        simf = simulate_vickrey_auction if auction_type == "Vickrey Auction" else simulate_generalized_auction

        for v in vals:
            agents.append(Agent(v, type=type, alpha=alpha))

        history = simf(agents, k, T)

        # Plot the history of the bids
        df = pd.DataFrame(history, columns=[f"Agent {a.v}" for a in agents])
        st.write("### Bids history")
        st.line_chart(df, x_label="Rounds", y_label="Bids")

        # Plot the weights of the agents
        index = np.arange(0, 1 + agents[0].eps, agents[0].eps).round(int(np.log10(1 / agents[0].eps)))
        df = pd.DataFrame(index=index)
        for a in agents:
            df[f"Agent {a.v}"] = 0
            df.loc[a.bids, f"Agent {a.v}"] = 1000 * a.weights / np.sum(a.weights)

        # st.area_chart(df, x_label="Bids", y_label="Weights")
        st.write("### Weights of the agents")
        st.bar_chart(df, x_label="Bids", y_label="Weights")
