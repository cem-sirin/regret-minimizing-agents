import streamlit as st

from agents import Auction
from plot import plot_bids, plot_weights


TYPE_MAP = {"First Price Auction": "fpa", "Second Price Auction": "spa"}

# 1. Introduction
st.title("Regret Minimizing Agents")

st.write(
    "This is a simple web app to simulate auctions with regret minimizing agents, using this "
    "implementation of the paper: https://arxiv.org/abs/2110.11855. An agent updates its weight by"
)

ltx = """
\\begin{align}
w_{i,t+1} &= w_{i,t} (1 + \eta \cdot u_{i,t}) \\\\
u_{i,t} &= v(x_{i,t}) - \\lambda \cdot p(x_{i,t}) \\\\
\\end{align}
"""
st.latex(ltx)
st.write(
    """
where:
- $w_{i,t}$ is the weight,
- $u_{i,t}$ is the utility,
- $x_{i,t}$ is the allocation of bid slot $i$ at time $t$,
- $v$ and $p$ are the value and payment functions respectively.
"""
)


# 2. Setting up the simulation
st.header("Configuration")

# Column 1
c1 = st.columns(3)

note = "Note: The input value of $$\epsilon$$ and $$\eta$$ parameters are the powers of 10. For example, if you want to set $$\epsilon = 10^{-2}$$, your input should be -2."
st.info(note)

λ = c1[0].number_input("Hybrid objective parameter ($$\lambda$$)", 0.0, 1.0, 1.0, 0.1)
# t = c1[0].selectbox("Auction Type", ["Second Price Auction", "First Price Auction"])
tau = c1[0].number_input("Overbidding factor ($$\\tau$$)", 0.0, 2.0, 1.0, 0.1)
α = c1[1].number_input("Reserve Price (α)", 0.0, 1.0, 0.10, 0.01)

eps = c1[2].number_input("Granularity of bids ($$\epsilon$$)", -10, -1, -3, 1)
eps = 10 ** (eps)
eta = c1[2].number_input("Learning Rate ($$\eta$$)", -10, -1, -2, 1)
eta = 10 ** (eta)

k = 1

c2 = st.columns(2)
# T = c2[0].slider("Number of Rounds", 1, 5000, 500)
auction_type = c2[0].selectbox("Auction Type", ["Second Price Auction", "First Price Auction"])

# Create a card for each agent
num_agents = c2[1].slider("Number of Agents", k + 1, 5, 3)

agent_columns = st.columns(num_agents)
vals = []
for i in range(num_agents):
    agent_columns[i].write(f"Agent {i+1}")
    vals.append(agent_columns[i].slider(f"$$v_{i+1}$$", α, 1.0, 0.5))

# Add button to adjust number of items
k = c1[1].number_input("Number of Items ($$k$$)", 1, num_agents - 1, 1)

T = st.slider("Number of Rounds", 1, 5000, 500)

agent_args = {"lam": λ, "eps": eps, "eta": eta, "tau": tau}

# Add button to start the simulation
if st.button("Start Simulation"):
    if any(v < α for v in vals):
        st.write("Values of agents must be greater than or equal to the reserve price.")
    else:

        auction = Auction(type=TYPE_MAP[auction_type], n=num_agents, v_list=vals, k=k, alpha=α, agent_args=agent_args)

        history = auction.simulate(T=T)
        ts_chart = plot_bids(history, auction)
        weights_chart = plot_weights(auction)

        st.altair_chart(ts_chart, use_container_width=True)
        st.altair_chart(weights_chart, use_container_width=True)
