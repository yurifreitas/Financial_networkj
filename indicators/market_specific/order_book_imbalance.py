import numpy as np

def compute(order_book: dict, depth=10) -> float:
    bids = order_book['bids'][:depth]
    asks = order_book['asks'][:depth]
    bid_volume = np.sum([q for _, q in bids])
    ask_volume = np.sum([q for _, q in asks])
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
    return float(imbalance)
