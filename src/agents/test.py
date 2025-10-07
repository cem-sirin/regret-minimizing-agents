import numpy as np
from rich import print

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
k = 3
eps = 0.01
bids = np.arange(0, 1, eps)  # slots
# print("bids", bids)
print("len(bids)", len(bids))

arr = np.random.rand(5)
arr.sort()
print("arr", arr)
print("arr.shape", arr.shape)

# top_k = arr[-(k + 1) : -1]
top_k = arr[-k:]
top_2nd_k = arr[-(k + 1) : -1]

print("top_k", top_k)
print("top_2nd_k", top_2nd_k)

cumsum = np.insert(np.flip(top_2nd_k[1:]).cumsum(), 0, 0)
print("cumsum", cumsum)


pass_count = (bids[..., np.newaxis] <= top_k[:-1]).sum(axis=-1)
print("pass_count", pass_count)
payment = cumsum[pass_count]


# From the index of the cheapest second price, to the maximum bid, we should add +eps to the payment
s = np.searchsorted(bids, arr[-k], side="right")
t = np.searchsorted(bids, arr[-1], side="right")
print(f"arr[-k]={arr[-k]}, arr[-1]={arr[-1]}")
print(f"s={s}, t={t}, bid_st={bids[s:t]}")
payment[s:t] += bids[s:t]

# print("pass_count", pass_count)
print("payment", payment)


x = np.digitize(bids, top_k, right=True)
print("x.shape", x.shape)


x = (bids[..., np.newaxis] <= top_k).sum(axis=-1)
print("x", x)
