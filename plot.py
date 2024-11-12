import pandas as pd
import altair as alt
import numpy as np


def plot_bids(history, auction):
    df = pd.DataFrame(history)
    df.columns = [f"Bidder {i+1} v={a.v.max()}" for i, a in enumerate(auction.bidders)]

    # Plot time series of bids using Altair
    # The columns should be "Bidder", "Bid", and "Round"
    df["Round"] = df.index
    df_long = df.melt(var_name="Bidder", value_name="Bid", id_vars=["Round"])

    # Calculate EWMA
    df_long["EWMA"] = df_long.groupby("Bidder")["Bid"].transform(lambda x: x.ewm(span=50).mean())

    # Create Altair chart with smoothed line and low opacity for noisy data
    chart = alt.Chart(df_long).properties(title="Bid Simulation")

    # Noisy data
    noisy_line = chart.mark_line(opacity=0.2).encode(x="Round:Q", y="Bid:Q", color="Bidder:N")

    # Smoothed line
    smoothed_line = chart.mark_line(color="black").encode(x="Round:Q", y="EWMA:Q", color="Bidder:N")

    chart = smoothed_line + noisy_line

    chart = chart.encode(y=alt.Y("Bid:Q", axis=alt.Axis(title="Bid")))  # y axis label
    return chart


def plot_weights(auction):
    eps = auction.bidders[0].eps
    index = np.arange(0, 1 + eps, eps)
    index = (index / eps).round().astype(int)

    df = pd.DataFrame(index=index)
    for i, a in enumerate(auction.bidders):
        x, y = a.bids, a.weights
        x = (x / eps).round().astype(int)
        df.loc[x, f"Agent {i} v={a.v.max()}"] = y

    df.index = (index * eps).round(3)
    df = df.reset_index()
    df = df.melt(id_vars="index", var_name="Agent", value_name="Weight")
    chart = (
        alt.Chart(df).mark_bar().encode(x="index:Q", y="Weight:Q", color="Agent:N").properties(title="Agent Weights")
    )
    return chart


if __name__ == "__main__":
    # Example usage (replace with your actual Auction and simulation)
    from agents import Auction

    auction = Auction(type="spa", n=3, v_list=[0.5, 0.6, 0.7, 1.0], k=2, alpha=0.1, agent_args={"eps": 1e-3})
    history = auction.simulate(T=1001)
    plot_bids(history)
