import altair as alt
import numpy as np
import pandas as pd


def plot_bids(history, auction):
    df = pd.DataFrame(history)
    df.columns = [f"Agents {i+1} v={a.v.max()}" for i, a in enumerate(auction.bidders)]

    # Plot time series of bids using Altair
    # The columns should be "Agents", "Bid", and "Round"
    df["Round"] = df.index
    df_long = df.melt(var_name="Agents", value_name="Bid", id_vars=["Round"])

    # Calculate EWMA
    df_long["EWMA"] = df_long.groupby("Agents")["Bid"].transform(lambda x: x.ewm(span=50).mean())

    # Create Altair chart with smoothed line and low opacity for noisy data
    chart = alt.Chart(df_long).properties(title="Bid Simulation")

    # Noisy data
    noisy_line = chart.mark_line(opacity=0.2).encode(x="Round:Q", y="Bid:Q", color="Agents:N")

    # Smoothed line
    smoothed_line = chart.mark_line(color="black").encode(x="Round:Q", y="EWMA:Q", color="Agents:N")

    # Add the reserve price as a vertical line
    df_reserve = pd.DataFrame({"alpha": [auction.alpha], "Agents": ["Reserve Price"]})
    reserve_line = (
        alt.Chart(df_reserve)
        .mark_rule(
            strokeDash=[5, 5],
            strokeWidth=2.5,
        )
        .encode(y="alpha:Q", color="Agents:N")
    )

    chart = smoothed_line + noisy_line + reserve_line
    # y axis label
    chart = chart.encode(y=alt.Y("Bid:Q", axis=alt.Axis(title="Bid")))
    return chart


def plot_weights(auction):
    eps = auction.bidders[0].eps

    # We need to first find the maximum bid value
    max_bid = max(a.bids[-1] for a in auction.bidders)
    index = np.arange(0, max_bid + eps, eps)
    index = (index / eps).round().astype(int)

    df = pd.DataFrame(index=index)
    for i, a in enumerate(auction.bidders):
        x, y = a.bids, a.weights
        x = (x / eps).round().astype(int)
        df.loc[x, f"Agents {i} v={a.v.max()}"] = y

    df.index = (index * eps).round(3)
    df = df.reset_index()
    df = df.melt(id_vars="index", var_name="Agents", value_name="Weight")
    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.5)
        .encode(
            x="index:Q",
            y=alt.Y("Weight:Q").stack(None),
            color="Agents:N",
        )
        .properties(title="Agent Weights")
    )

    # Add the reserve price as a horizontal line
    df_reserve = pd.DataFrame({"alpha": [auction.alpha], "Agents": ["Reserve Price"]})
    reserve_line = (
        alt.Chart(df_reserve)
        .mark_rule(
            strokeDash=[5, 5],
            strokeWidth=2.5,
        )
        .encode(x="alpha:Q", color="Agents:N")
    )

    chart = chart + reserve_line

    # x axis label
    chart = chart.encode(x=alt.X("index:Q", axis=alt.Axis(title="Bid")))

    return chart


if __name__ == "__main__":
    # Example usage (replace with your actual Auction and simulation)
    from agents import Auction

    auction = Auction(type="spa", n=3, v_list=[0.5, 0.6, 0.7, 1.0], k=2, alpha=0.1, agent_args={"eps": 1e-3})
    history = auction.simulate(T=1001)
    plot_bids(history)
