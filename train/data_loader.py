import numpy as np
from pong import Pong
from render import Renderer

def collect_buffer(N: int,
                   T: int = 300,
                   seed: int | None = None,
                   print_every: int = 50) -> dict:
    """Collect a replay buffer of N episodes, each with T timesteps.
       Returns dict with crops, positions, and actions."""
    rng = np.random.default_rng(seed)
    env = Pong()
    renderer = Renderer(env.settings)

    paddle_left_x = env.settings["paddle_left_x"]
    paddle_right_x = env.settings["paddle_right_x"]
    score_center = env.settings["score_center"]
    resolution = env.settings["resolution"]

    crops_buf = None
    pos_buf = None
    left_a_buf = np.empty((N, T), dtype=np.int32)
    right_a_buf = np.empty((N, T), dtype=np.int32)

    for ep in range(N):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        env.step(left_action=None, right_action=None)

        for t in range(T):
            left_t, right_t, ball_t, score_t = renderer.render_crops(state=env.state, prev_state=env.prev_state)
            crops = np.stack([left_t, right_t, ball_t, score_t], axis=0)
            crops = np.transpose(crops, (0, 3, 1, 2))

            pos = np.array([[paddle_left_x, env.state["paddle_left_y"]],
                            [paddle_right_x, env.state["paddle_right_y"]],
                            [env.state["ball_x"], env.state["ball_y"]],
                            [score_center[0], score_center[1]]], dtype=np.float32)
            pos = pos / resolution * 2.0 - 1.0

            _, info = env.step(left_action=None, right_action=None)

            if crops_buf is None:
                crops_buf = np.empty((N, T) + crops.shape, dtype=np.uint8)
                pos_buf = np.empty((N, T) + pos.shape, dtype=np.float32)

            crops_buf[ep, t] = crops
            pos_buf[ep, t] = pos
            left_a_buf[ep, t] = int(info["left_action"])
            right_a_buf[ep, t] = int(info["right_action"])

        if ((ep + 1) % print_every) == 0 or ep == 0:
            print("episode:", ep + 1)

    buffer = {"crops": crops_buf,      # (N, T, K, 3, H_img, W_img) uint8
              "pos": pos_buf,          # (N, T, K, 2) float32, normalized to [-1, 1]
              "left_a": left_a_buf,    # (N, T) int32
              "right_a": right_a_buf}  # (N, T) int32

    return buffer