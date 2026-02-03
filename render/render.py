import numpy as np

# Rendering Parameters
DISPLAY_SETTINGS = {
    "digit_patterns": {0: [(0, 0, 3, 1), (0, 0, 1, 5), (2, 0, 1, 5), (0, 4, 3, 1)],
                       1: [(1, 0, 1, 5)],
                       2: [(0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1), (2, 2, 1, 3), (0, 0, 1, 3)],
                       3: [(2, 0, 1, 5), (0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1)],
                       4: [(2, 0, 1, 5), (0, 2, 3, 1), (0, 2, 1, 3)],
                       5: [(0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1), (2, 0, 1, 3), (0, 2, 1, 3)],
                       6: [(0, 0, 1, 5), (0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1), (2, 0, 1, 3)],
                       7: [(0, 4, 3, 1), (2, 0, 1, 5)],
                       8: [(0, 0, 1, 5), (2, 0, 1, 5), (0, 0, 3, 1), (0, 2, 3, 1), (0, 4, 3, 1)],
                       9: [(2, 0, 1, 5), (0, 2, 3, 1), (0, 4, 3, 1), (0, 2, 1, 3)]},
        
    "colors": {"paddle_left": np.array([0, 0, 255], dtype=np.float32),      
               "paddle_right": np.array([0, 255, 0], dtype=np.float32),  
               "ball": np.array([255, 0, 0], dtype=np.float32),     
               "paddle_left_prev": np.array([0, 0, 89], dtype=np.float32),  
               "paddle_right_prev": np.array([0, 89, 0], dtype=np.float32),
               "ball_prev": np.array([0, 190, 190], dtype=np.float32),   
               "score": np.array([128, 128, 128], dtype=np.float32)}}


# Rendering Helpers
def _draw_rect(img, x, y, w, h, color):
    """Draw a filled rectangle. (x, y) is bottom-left corner."""
    buffer_size = img.shape[0]
    
    r0 = int(np.floor(buffer_size - (y + h)))
    r1 = int(np.ceil(buffer_size - y))
    c0 = int(np.floor(x))
    c1 = int(np.ceil(x + w))

    r0 = max(0, min(buffer_size, r0))
    r1 = max(0, min(buffer_size, r1))
    c0 = max(0, min(buffer_size, c0))
    c1 = max(0, min(buffer_size, c1))

    if r1 > r0 and c1 > c0:
        img[r0:r1, c0:c1] = color

def _draw_circle(img, x, y, r, color):
    """Draw a filled circle. (x, y) is center."""
    buffer_size = img.shape[0]

    x_min, x_max = int(np.floor(x - r)), int(np.ceil(x + r))
    y_min, y_max = int(np.floor(y - r)), int(np.ceil(y + r))

    row_min = max(0, int(np.floor(buffer_size - y_max)))
    row_max = min(buffer_size, int(np.ceil(buffer_size - y_min)))
    col_min = max(0, x_min)
    col_max = min(buffer_size, x_max)

    if row_max <= row_min or col_max <= col_min:
        return

    rows = np.arange(row_min, row_max, dtype=np.float32)
    cols = np.arange(col_min, col_max, dtype=np.float32)

    pixel_x = cols[None, :] + 0.5                     
    pixel_y = (buffer_size - 1 - rows[:, None]) + 0.5   

    dx = pixel_x - x
    dy = pixel_y - y
    mask = (dx*dx + dy*dy) <= r*r  

    img[row_min:row_max, col_min:col_max][mask] = color

def _draw_digit(img, x, y, digit, digit_scale, color, digit_patterns):
    """Draw a score digit using digit_patterns."""
    if digit not in digit_patterns:
        return
    for dx, dy, dw, dh in digit_patterns[digit]:
        _draw_rect(img,
                   x + dx * digit_scale,
                   y + dy * digit_scale,
                   dw * digit_scale,
                   dh * digit_scale,
                   color)

def _downsample(img, crop_size: int, upscale_factor: int):
    """Box-downsample high-res buffer to output size."""
    if upscale_factor == 1:
        return np.clip(np.round(img), 0, 255).astype(np.uint8)
    img = img.reshape(crop_size, upscale_factor, crop_size, upscale_factor, 3).mean(axis=(1, 3))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# Rendering Class
class Renderer:
    def __init__(self, settings, display_settings=None):
        self.settings = settings
        self.display_settings = display_settings if display_settings is not None else DISPLAY_SETTINGS
        self._buffer_cache = {}

    def _get_buffer(self, buffer_size, idx):
        cache_key = (buffer_size, idx)
        
        if cache_key in self._buffer_cache:
            buf = self._buffer_cache[cache_key]
        else:
            buf = np.zeros((buffer_size, buffer_size, 3), dtype=np.float32)
            self._buffer_cache[cache_key] = buf
            
        buf.fill(0.0)
        return buf

    def clear_cache(self):
        self._buffer_cache.clear()

    def render_crops(self, state, prev_state):
        """Return (left_paddle, right_paddle, ball, score) crops as uint8 images."""
        crop_size = self.settings["crop_size"]
        upscale_factor = self.settings["upscale_factor"]

        buffer_size = crop_size * upscale_factor
        center = buffer_size / 2.0

        digit_patterns = self.display_settings["digit_patterns"]
        colors = self.display_settings["colors"]

        paddle_w = self.settings["paddle_width"] * upscale_factor
        paddle_h = self.settings["paddle_height"] * upscale_factor
        ball_r = self.settings["ball_radius"] * upscale_factor
        digit_scale = self.settings["score_scale"] * upscale_factor

        paddle_x = center - paddle_w / 2
        paddle_y = center - paddle_h / 2

        # Left paddle
        left_buffer = self._get_buffer(buffer_size, idx=0)
        if prev_state is not None:
            paddle_left_y_prev = paddle_y + (prev_state["paddle_left_y"] - state["paddle_left_y"]) * upscale_factor
            _draw_rect(left_buffer, paddle_x, paddle_left_y_prev, paddle_w, paddle_h, colors["paddle_left_prev"])    
        _draw_rect(left_buffer, paddle_x, paddle_y, paddle_w, paddle_h, colors["paddle_left"])

        # Right paddle
        right_buffer = self._get_buffer(buffer_size, idx=1)
        if prev_state is not None:
            paddle_right_y_prev = paddle_y + (prev_state["paddle_right_y"] - state["paddle_right_y"]) * upscale_factor
            _draw_rect(right_buffer, paddle_x, paddle_right_y_prev, paddle_w, paddle_h, colors["paddle_right_prev"]) 
        _draw_rect(right_buffer, paddle_x, paddle_y, paddle_w, paddle_h, colors["paddle_right"])

        # Ball
        ball_buffer = self._get_buffer(buffer_size, idx=2)
        if prev_state is not None:
            ball_x_prev = center + (prev_state["ball_x"] - state["ball_x"]) * upscale_factor
            ball_y_prev = center + (prev_state["ball_y"] - state["ball_y"]) * upscale_factor
            _draw_circle(ball_buffer, ball_x_prev, ball_y_prev, ball_r, colors["ball_prev"])
        _draw_circle(ball_buffer, center, center, ball_r, colors["ball"])

        # Score
        score_buffer = self._get_buffer(buffer_size, idx=3)
        _draw_digit(score_buffer, center - 4.5 * digit_scale, center - 2.5 * digit_scale,
                    int(state["score_left"]), digit_scale, colors["score"], digit_patterns)
        _draw_rect(score_buffer, center - 0.5 * digit_scale, center - 1.5 * digit_scale,
                   digit_scale, digit_scale, colors["score"])
        _draw_rect(score_buffer, center - 0.5 * digit_scale, center + 0.5 * digit_scale,
                   digit_scale, digit_scale, colors["score"])
        _draw_digit(score_buffer, center + 1.5 * digit_scale, center - 2.5 * digit_scale,
                    int(state["score_right"]), digit_scale, colors["score"], digit_patterns)

        return (_downsample(left_buffer, crop_size, upscale_factor),
                _downsample(right_buffer, crop_size, upscale_factor),
                _downsample(ball_buffer, crop_size, upscale_factor),
                _downsample(score_buffer, crop_size, upscale_factor))

    def reconstruct_frame(self, crops, state):
        """Composite crop images into a full frame."""
        resolution = self.settings["resolution"]
        crop_size = crops[0].shape[0]
        half = crop_size // 2

        # Back to front: score, left paddle, right paddle, ball
        z_order = [3, 0, 1, 2]
        positions = [(self.settings["paddle_left_x"], state["paddle_left_y"]),
                     (self.settings["paddle_right_x"], state["paddle_right_y"]),
                     (state["ball_x"], state["ball_y"]),
                     (self.settings["score_center"][0], self.settings["score_center"][1])]
        
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        for i in z_order:
            crop = crops[i]
            x, y = positions[i]
            
            col_center = int(round(x))
            row_center = int(round(resolution - y))
            
            dst_r1 = max(0, row_center - half)
            dst_r2 = min(resolution, row_center + half)
            dst_c1 = max(0, col_center - half)
            dst_c2 = min(resolution, col_center + half)
            
            src_r1 = half - (row_center - dst_r1)
            src_r2 = src_r1 + (dst_r2 - dst_r1)
            src_c1 = half - (col_center - dst_c1)
            src_c2 = src_c1 + (dst_c2 - dst_c1)
            
            if dst_r2 > dst_r1 and dst_c2 > dst_c1:
                src_region = crop[src_r1:src_r2, src_c1:src_c2]
                mask = src_region.any(axis=-1)
                frame[dst_r1:dst_r2, dst_c1:dst_c2][mask] = src_region[mask]
        
        return frame
