import numpy as np
from data_loader import collect_buffer

if __name__ == "__main__":
    buffer = collect_buffer(N=512, T=300, print_every=50)
    np.savez("pong_buffer.npz", **buffer)