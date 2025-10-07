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

# Presets
st.sidebar.subheader("Presets")
preset_options = ["Custom", "SPA with equal values", "SPA with seller"]
selected_preset = st.sidebar.selectbox("Choose a preset", preset_options)


# Define preset configurations
def apply_preset(preset_name):
    if preset_name == "SPA with equal values":
        return {
            "auction_type": "Second Price Auction",
            "num_agents": 3,
            "k": 1,
            "α": 0.1,
            "include_seller": False,
            "λ": 1.0,
            "tau": 1.0,
            "eps": -3,
            "eta": -2,
            "T": 5000,
            "v_list": [0.5, 0.5, 0.5],
        }
    elif preset_name == "SPA with seller":
        return {
            "auction_type": "Second Price Auction",
            "num_agents": 3,
            "k": 1,
            "α": 0.1,
            "include_seller": True,
            "λ": 1.0,
            "tau": 1.0,
            "eps": -3,
            "eta": -2,
            "T": 5000,
            "v_list": [1.0, 0.5, 0.5],  # First agent has value 1
            "seller_cost": 0.05,
        }
    else:
        return {}  # Custom - use current values


# Apply preset if not custom
preset_config = apply_preset(selected_preset)

st.sidebar.info(
    r"Note: The input value of $$\epsilon$$ and $$\eta$$ parameters are the powers of 10. For example, if you want to set $$\epsilon = 10^{-2}$$, your input should be -2."
)

if selected_preset != "Custom":
    st.success(f"🎯 **Active Preset:** {selected_preset}")
    st.info("👈 You can modify parameters below or choose another preset")
else:
    st.info(
        "👈 Configure auction parameters in the sidebar and click 'Start Simulation' below"
    )

# Auction parameters
st.sidebar.subheader("Auction Parameters")
auction_type = st.sidebar.selectbox(
    "Auction Type",
    ["Second Price Auction", "First Price Auction"],
    index=["Second Price Auction", "First Price Auction"].index(
        preset_config.get("auction_type", "Second Price Auction")
    )
    if selected_preset != "Custom"
    else 0,
)
num_agents = st.sidebar.slider(
    "Number of Agents", 2, 5, preset_config.get("num_agents", 2)
)
k = st.sidebar.slider(r"Number of Items ($$k$$)", 1, 4, preset_config.get("k", 1))
α = st.sidebar.slider(
    r"Reserve Price (α)", 0.0, 1.0, preset_config.get("α", 0.10), 0.01
)
include_seller = st.sidebar.checkbox(
    "Include Seller", preset_config.get("include_seller", False)
)

# Agent parameters
st.sidebar.subheader("Agent Parameters")
λ = st.sidebar.slider(
    r"Hybrid objective parameter ($$\lambda$$)",
    0.0,
    1.0,
    preset_config.get("λ", 1.0),
    0.1,
)
tau = st.sidebar.slider(
    r"Overbidding factor ($$\tau$$)", 0.0, 2.01, preset_config.get("tau", 1.0), 0.1
)

# Seller parameters (only shown if seller is included)
seller_cost = preset_config.get("seller_cost", 0.0)  # Default value
if include_seller:
    st.sidebar.subheader("Seller Parameters")
    seller_cost = st.sidebar.slider(
        r"Seller Cost per Item", 0.0, 1.0, seller_cost, 0.01
    )

eps = st.sidebar.slider(
    r"Granularity of bids ($$\epsilon$$)", -10, -1, preset_config.get("eps", -3), 1
)
eps = 10 ** (eps)
eta = st.sidebar.slider(
    r"Learning Rate ($$\eta$$)", -10, -1, preset_config.get("eta", -2), 1
)
eta = 10 ** (eta)

# Agent values
st.sidebar.subheader("Agent Values")
v_list = []
preset_v_list = preset_config.get("v_list", [])
for i in range(num_agents):
    default_value = preset_v_list[i] if i < len(preset_v_list) else 0.5
    v_list.append(
        st.sidebar.slider(f"Agent {i + 1} Value ($$v_{i + 1}$$)", α, 1.0, default_value)
    )

# Simulation parameters
st.sidebar.subheader("Simulation")
T = st.sidebar.slider(
    "Number of Rounds", 0, 50_000, preset_config.get("T", 5000), step=100
)
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
