# Auction Simulation App for Regret-Minimizing Agents 

Established the foundation from the paper [Auctions Between Regret-Minimizing Agents](https://arxiv.org/abs/2110.11855) by Yoav Kolombus and Noam Nisan.

## Auction Types

### Generalized First Price Auction (fpa):
- Buyers pay their own bid

### Generalized Second Price Auction (spa): winners pay the bid that comes after them
- Buyers pay the price that comes after them

### Future implementations
- Uniform Price Auction (upa): winners pay the highest non-winning bid
- Vickrey-Clarke-Groves Auction (vcga): winners pay the highest bid that is greater than their own bid

## Coding Conventions

### Notation

- k [int]: number of items in the auction
- v [array(k+1,)]: array of values of the items in the auction
- lam [float]: hybrid objective function parameter
- eps [float]: granularity of bids (e.g., 0.001, 0.0001 etc.)
- N [int]: number of actions (i.e., number of bid prices, eps^-1)
- alpha [float]: reserve price (agents can't bid below this price)
- bids [array(N,)]: corresponding bids for each action (e.g., {0, 0.001, 0.002, ..., 0.999})
- x [array(N,)]: allocation of each action, values range from {0, 1, ..., k} indicating which slot action i would result to.
- p [array(N,)]: payments of each action
- eta [float]: learning rate of the multiplicative weights algorithm
- weights [array(N,)]: weights of each action

### Misc
- Everything is sorted in ascending order. The values of items must be item_1 < item_2 < ... < item_k.
- The allocation 0 means no items won. For example, if x = [0, 0, 1, 2], then bids 0 and 1 did not win any items, bid 2 won item 1, and bid 3 won item 2.
- In our work we will include the `seller` as a bidding agent, that can bid a reserve price. Therefore we will use the term `buyer` rather than using the term `bidder` for those who participate to pay for the auction item.