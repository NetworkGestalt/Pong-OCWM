import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def merge_crops(crops: list[np.ndarray], pad: int = 2) -> np.ndarray:
    """Horizontally concatenate object crops into a single image with padding between them."""
    pad = int(pad)
    crops = list(crops)
    h, w = crops[0].shape[:2]
    n = len(crops)

    # Canvas width: n crops + (n-1) padding gaps
    canvas = np.zeros((h, n * w + (n - 1) * pad, 3), dtype=np.uint8)

    # Place each crop side by side
    x = 0
    for i, c in enumerate(crops):
        canvas[:, x:x + w] = c
        x += w + (pad if i < n - 1 else 0)   # no padding after last crop

    return canvas


def create_animation(frames: list[np.ndarray], fps: int = 24, fig_scale: int = 5) -> FuncAnimation:
    """Create a matplotlib animation from a sequence of frames."""
    frame0 = frames[0]
    h, w = frame0.shape[:2]
    dpi = 100

    # Size figure to match frame aspect ratio
    fig = plt.figure(figsize=(fig_scale * w / dpi, fig_scale * h / dpi), dpi=dpi, frameon=False)

    # Fill entire figure with axes (no margins)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Initialize with first frame
    im = ax.imshow(frame0, interpolation="nearest", animated=True)

    def update(i):
        """Update function called each frame by FuncAnimation."""
        im.set_data(frames[i])
        return (im,)

    # Convert fps to milliseconds between frames
    interval_ms = int(round(1000 / fps))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    plt.close(fig)

    return anim
