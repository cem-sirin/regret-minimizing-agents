import streamlit as st

from src.agents.base import AuctionType
from src.auctions import Auction
from src.plots import plot_bids, plot_weights

st.set_page_config(page_title="Regret Minimizing Agents", layout="wide")
AUCTION_TYPE_MAP = {
    "First Price Auction": AuctionType.FPA,
    "Second Price Auction": AuctionType.SPA,
}

# 1. Introduction
st.title("Regret Minimizing Agents")

c0 = st.columns(2)

c0[0].write(
    "A simple web app to simulate auctions with regret minimizing agents, using this "
    "implementation in [Kolombus & Nisan, 2021](https://arxiv.org/abs/2110.11855)."
)

c0[0].write(
    "For an auction with $$n$$ agents and $$k$$ items, an agent updates its weight by"
)

ltx = """
\\begin{align}
u_{i,t} &= v(x_{i,t}) - \\lambda \cdot p(x_{i,t}) \\\\
w_{i,t+1} &= w_{i,t} (1 + \eta \cdot u_{i,t}) \\\\ 
\\end{align}
""".strip()
c0[0].latex(ltx)
c0[0].write("""
where:
- $u_{i,t}$, $w_{i,t}$, and $x_{i,t}$ are the utility, weight, and allocation of bid slot $i$ at time $t$ respectively,
- $v(\cdot)$ and $p(\cdot)$ are the value and payment functions respectively.
""")


# Column 1
c1 = c0[1].columns(3)

c0[1].info(
    "Note: The input value of $$\epsilon$$ and $$\eta$$ parameters are the powers of 10. For example, if you want to set $$\epsilon = 10^{-2}$$, your input should be -2."
)

λ = c1[0].number_input("Hybrid objective parameter ($$\lambda$$)", 0.0, 1.0, 1.0, 0.1)
tau = c1[0].number_input("Overbidding factor ($$\\tau$$)", 0.0, 2.01, 1.0, 0.1)
α = c1[1].number_input("Reserve Price (α)", 0.0, 1.0, 0.10, 0.01)

eps = c1[2].number_input("Granularity of bids ($$\epsilon$$)", -10, -1, -3, 1)
eps = 10 ** (eps)
eta = c1[2].number_input("Learning Rate ($$\eta$$)", -10, -1, -2, 1)
eta = 10 ** (eta)

k = 1

c2 = c0[1].columns(2)
auction_type = c2[0].selectbox(
    "Auction Type", ["Second Price Auction", "First Price Auction"]
)

# Create a card for each agent
num_agents = c2[1].slider("Number of Agents", k + 1, 5, 2)

agent_columns = c2[1].columns(num_agents)
v_list = []
for i in range(num_agents):
    agent_columns[i].write(f"Agent {i + 1}")
    v_list.append(agent_columns[i].slider(f"$$v_{i + 1}$$", α, 1.0, 0.5))

# Add button to adjust number of items
k = c1[1].number_input("Number of Items ($$k$$)", 1, num_agents - 1, 1)

T = c0[1].slider("Number of Rounds", 0, 50_000, 5_000, step=100)
agent_args = {"lam": λ, "eps": eps, "eta": eta, "tau": tau}

# Add button to start the simulation
if st.button("Start Simulation"):
    if any(v < α for v in v_list):
        st.write("Values of agents must be greater than or equal to the reserve price.")
    else:
        auction = Auction(
            auction_type=AUCTION_TYPE_MAP[auction_type],
            n=num_agents,
            v_list=v_list,
            k=k,
            alpha=α,
            agent_args=agent_args,
        )

        history = auction.simulate(T=T)
        ts_chart = plot_bids(history, auction)
        weights_chart = plot_weights(auction)

        st.altair_chart(ts_chart, use_container_width=True)
        st.altair_chart(weights_chart, use_container_width=True)
