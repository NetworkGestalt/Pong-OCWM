import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Animation Functions
def merge_crops(crops, pad: int = 2):
    """Horizontally concatenate object crops into one image."""
    pad = int(pad)
    crops = list(crops)

    h, w = crops[0].shape[:2]
    n = len(crops)

    canvas = np.zeros((h, n * w + (n - 1) * pad, 3), dtype=np.uint8)

    x = 0
    for i, c in enumerate(crops):
        canvas[:, x:x + w] = c
        x += w + (pad if i < n - 1 else 0)

    return canvas

def create_animation(frames, fps: int = 24, fig_scale: int = 5):
    """Create a matplotlib animation from a sequence of frames."""
    frame0 = frames[0]
    h, w = frame0.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(fig_scale * w / dpi, fig_scale * h / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(frame0, interpolation="nearest", animated=True)

    def update(i):
        im.set_data(frames[i])
        return (im,)

    interval_ms = int(round(1000 / fps))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    plt.close(fig)
    
    return anim
