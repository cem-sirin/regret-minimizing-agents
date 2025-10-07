import streamlit as st

from src.agents.base import AuctionType
from src.auctions import Auction, AuctionConfig
from src.plots import plot_bids, plot_weights

st.set_page_config(page_title="Regret Minimizing Agents")
AUCTION_TYPE_MAP = {
    "First Price Auction": AuctionType.FPA,
    "Second Price Auction": AuctionType.SPA,
}

# 1. Introduction
st.title("Regret Minimizing Agents")


st.write(
    "A simple web app to simulate auctions with regret minimizing agents, using this "
    "implementation in [Kolombus & Nisan, 2021](https://arxiv.org/abs/2110.11855)."
)

st.write(
    "For an auction with $$n$$ agents and $$k$$ items, an agent updates its weight by"
)


st.latex(r"""
\begin{align}
u_{i,t} &= v(x_{i,t}) - \lambda \cdot p(x_{i,t}) \\
w_{i,t+1} &= w_{i,t} (1 + \eta \cdot u_{i,t}) \\ 
\end{align}
""")
st.write(r"""
where:
- $u_{i,t}$, $w_{i,t}$, and $x_{i,t}$ are the utility, weight, and allocation of bid slot $i$ at time $t$ respectively,
- $v(\cdot)$ and $p(\cdot)$ are the value and payment functions respectively.
""")


# Sidebar Configuration
st.sidebar.header("Auction Configuration")

st.sidebar.info(
    r"Note: The input value of $$\epsilon$$ and $$\eta$$ parameters are the powers of 10. For example, if you want to set $$\epsilon = 10^{-2}$$, your input should be -2."
)

st.info(
    "👈 Configure auction parameters in the sidebar and click 'Start Simulation' below"
)

# Auction parameters
st.sidebar.subheader("Auction Parameters")
auction_type = st.sidebar.selectbox(
    "Auction Type", ["Second Price Auction", "First Price Auction"]
)
num_agents = st.sidebar.slider("Number of Agents", 2, 5, 2)
k = st.sidebar.slider(r"Number of Items ($$k$$)", 1, 4, 1)
α = st.sidebar.slider(r"Reserve Price (α)", 0.0, 1.0, 0.10, 0.01)
include_seller = st.sidebar.checkbox("Include Seller", False)

# Agent parameters
st.sidebar.subheader("Agent Parameters")
λ = st.sidebar.slider(r"Hybrid objective parameter ($$\lambda$$)", 0.0, 1.0, 1.0, 0.1)
tau = st.sidebar.slider(r"Overbidding factor ($$\tau$$)", 0.0, 2.01, 1.0, 0.1)

# Seller parameters (only shown if seller is included)
seller_cost = 0.0  # Default value
if include_seller:
    st.sidebar.subheader("Seller Parameters")
    seller_cost = st.sidebar.slider(r"Seller Cost per Item", 0.0, 1.0, 0.0, 0.01)

eps = st.sidebar.slider(r"Granularity of bids ($$\epsilon$$)", -10, -1, -3, 1)
eps = 10 ** (eps)
eta = st.sidebar.slider(r"Learning Rate ($$\eta$$)", -10, -1, -2, 1)
eta = 10 ** (eta)

# Agent values
st.sidebar.subheader("Agent Values")
v_list = []
for i in range(num_agents):
    v_list.append(
        st.sidebar.slider(f"Agent {i + 1} Value ($$v_{i + 1}$$)", α, 1.0, 0.5)
    )

# Simulation parameters
st.sidebar.subheader("Simulation")
T = st.sidebar.slider("Number of Rounds", 0, 50_000, 5_000, step=100)
agent_args = {"lam": λ, "eps": eps, "eta": eta, "tau": tau}
if include_seller:
    agent_args["seller_cost"] = seller_cost


# Add button to start the simulation
if st.button("Start Simulation"):
    if any(v < α for v in v_list):
        st.write("Values of agents must be greater than or equal to the reserve price.")
    elif k >= num_agents:
        st.write("Number of items must be less than number of agents.")
    else:
        auction_config = AuctionConfig(
            auction_type=AUCTION_TYPE_MAP[auction_type],
            n=num_agents,
            v_list=v_list,
            k=k,
            alpha=α,
            agent_args=agent_args,
            include_seller=include_seller,
        )
        auction = Auction(auction_config)

        history = auction.simulate(T=T)
        ts_chart = plot_bids(history, auction)
        weights_chart = plot_weights(auction)

        st.altair_chart(ts_chart, use_container_width=True)
        st.altair_chart(weights_chart, use_container_width=True)
