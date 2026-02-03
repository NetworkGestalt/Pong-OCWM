import numpy as np
from pong import Pong
from render import Renderer
from animate import merge_crops, create_animation

if __name__ == "__main__":
    T = 300
    env = Pong()
    env.reset(seed=np.random.randint(1_000_000))
    renderer = Renderer(env.settings)
    
    object_crops = []
    full_frames = []
    
    for _ in range(T):
        state, info = env.step()
        left, right, ball, score = renderer.render_crops(state=state, prev_state=env.prev_state)
        object_crops.append(merge_crops([left, right, ball, score]))
        full_frame = renderer.reconstruct_frame(crops=(left, right, ball, score), state=state)
        full_frames.append(full_frame)
    
    anim = create_animation(full_frames)
    HTML(anim.to_jshtml())

    # To save as .gif (requires Pillow package):
    # anim.save("pong.gif", writer="pillow", fps=24))
