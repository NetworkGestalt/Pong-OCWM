import numpy as np
from typing import Optional

def create_settings() -> dict:
    """Create and return the default game configuration dictionary."""
    settings = {
        # Game canvas & rendering
        "resolution": 64,           # game canvas side length in pixels
        "border_size": 2,           # wall thickness
        "crop_size": 20,            # object crop dimensions
        "upscale_factor": 4,        # internal render buffer multiplier

        # Ball
        "ball_vel_magn": 2.0,       # speed (pixels per step)
        "ball_vel_dir_noise": 0.0,  # per-step direction jitter
        "ball_bounce_noise": 0.1,   # direction jitter on bounce

        # Paddles
        "ball_radius": 2.4,
        "paddle_height": 12,
        "paddle_width": 4,
        "action_step_size": 2.0,    # pixels per action
        "action_step_noise": 0.05,  # movement jitter

        # Score
        "score_scale": 2.0, 
        "max_points": 5, 

        # AI controller
        "ai_decay": 0.95,           # bias persistence (AR(1) coefficient)
        "ai_bias_noise": 1.7,       # bias shock magnitude
        "prob_rand_action": 0.05,   # chance of random action per step
        "ai_dead_zone": 2.0,        # ignore ball if within this y-distance (prevents oscillations)
    }

    # Left paddle position limits    
    settings["paddle_left_x"] = 3 + settings["border_size"]
    settings["paddle_left_y_min"] = settings["border_size"] + (settings["paddle_height"] / 2.0)
    settings["paddle_left_y_max"] = settings["resolution"] - settings["border_size"] - (settings["paddle_height"] / 2.0)

    # Right paddle position limits (mirrored)
    settings["paddle_right_x"] = settings["resolution"] - settings["paddle_left_x"]
    settings["paddle_right_y_min"] = settings["border_size"] + (settings["paddle_height"] / 2.0)
    settings["paddle_right_y_max"] = settings["resolution"] - settings["border_size"] - (settings["paddle_height"] / 2.0)

    # Ball position limits
    settings["ball_x_min"] = settings["border_size"]
    settings["ball_x_max"] = settings["resolution"] - settings["border_size"]
    settings["ball_y_max"] = settings["resolution"] - settings["ball_radius"] - settings["border_size"]
    settings["ball_y_min"] = settings["ball_radius"] + settings["border_size"]

    # Ball boundaries for scoring and spawning
    settings["ball_x_min_point"] = settings["ball_radius"] + settings["border_size"]
    settings["ball_x_max_point"] = settings["resolution"] - settings["ball_radius"] - settings["border_size"]
    settings["ball_x_max_sample"] = settings["paddle_right_x"] - (settings["paddle_width"] / 2.0) - settings["ball_radius"]
    settings["ball_x_min_sample"] = settings["paddle_left_x"] + (settings["paddle_width"] / 2.0) + settings["ball_radius"]
    settings["ball_pos_noise"] = settings["ball_vel_magn"] * 0.1

    # Center coordinates for resets and rendering
    settings["center_point_x"] = settings["resolution"] / 2.0
    settings["center_point_y"] = settings["resolution"] / 2.0
    settings["score_center"] = (int(settings["resolution"] / 2), int(settings["resolution"] * 0.88))

    return settings


class Pong:
    ACTIONS = {-1: "down", 0: "still", 1: "up"} 

    def __init__(self, settings: Optional[dict] = None) -> None:
        self.settings = create_settings() if settings is None else settings
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.state = None
        self.prev_state = None
        self.ai_bias_left = 0.0
        self.ai_bias_right = 0.0

    def reset(self, seed: int = 0) -> dict:
        """Reset the environment to a random initial state."""
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.prev_state = None
        self.ai_bias_left = 0.0
        self.ai_bias_right = 0.0

        init_state = {}
        init_state["ball_x"] = self.rng.uniform(self.settings["ball_x_min_sample"], self.settings["ball_x_max_sample"])
        init_state["ball_y"] = self.rng.uniform(self.settings["ball_y_min"], self.settings["ball_y_max"])
        init_state["ball_vel_magn"] = self.settings["ball_vel_magn"]
        init_state["ball_vel_dir"] = self.rng.uniform(0.0, 2.0 * np.pi)
        init_state["paddle_left_y"] = self.rng.uniform(self.settings["paddle_left_y_min"], self.settings["paddle_left_y_max"])
        init_state["paddle_right_y"] = self.rng.uniform(self.settings["paddle_right_y_min"], self.settings["paddle_right_y_max"])
        init_state["score_left"] = int(self.rng.integers(self.settings["max_points"]))
        init_state["score_right"] = int(self.rng.integers(self.settings["max_points"]))

        self.state = init_state
        return init_state

    # ------------------------------
    # AI Controller
    # ------------------------------
    
    def _get_ai_action(self, paddle_y: float, ball_y: float, current_bias: float) -> tuple[int, float]:
        """Compute AI action based on perceived ball position. Uses an AR(1) process to emulate imperfect perception/human error,
        in addition to a small probability of taking a random action."""

        # Update perception bias: bias_t = decay * bias_{t-1} + noise
        decay = self.settings["ai_decay"]
        noise_scale = self.settings["ai_bias_noise"]
        new_bias = decay * current_bias + noise_scale * self.rng.standard_normal()
        
        if self.rng.uniform() < self.settings["prob_rand_action"]:
            action = int(self.rng.choice([-1, 0, 1]))
        else:
            perceived_ball_y = ball_y + new_bias
            diff = perceived_ball_y - paddle_y

            # Dead zone: prevents oscillation when paddle is close to ball in the y-axis
            if abs(diff) < self.settings["ai_dead_zone"]:
                action = 0  
            elif diff > 0:
                action = 1    # paddle moves up
            else:
                action = -1   # paddle moves down
        
        return action, new_bias

    def _apply_paddle_action(self, paddle_y: float, action: int) -> float:
        """Apply action to paddle position with movement noise."""
        step = action * self.settings["action_step_size"]
        noise = self.rng.standard_normal() * self.settings["action_step_noise"]
        return paddle_y + step + noise

    # ------------------------------
    # Dynamics Helpers
    # ------------------------------
    
    def _mod_angle(self, angle: float) -> float:
        """Normalize angle to [0, 2π) range."""
        return angle % (2.0 * np.pi)

    def _angle_flip(self, angle: float, axis: str = "y") -> float:
        """Reflect angle across the specified axis ('x' for horizontal, 'y' for vertical)."""
        if axis == "x":
            return self._mod_angle(np.pi - angle)
        else:
            return self._mod_angle(-angle)

    def _ball_collision(self, paddle_tag: str, new_state: dict, prev_state: dict) -> bool:
        """Check if the ball collides with the specified paddle based on whether their bounding boxes overlap."""
        paddle_x = self.settings[paddle_tag + "_x"]
        paddle_y = prev_state[paddle_tag + "_y"]
        paddle_height = self.settings["paddle_height"]
        paddle_width = self.settings["paddle_width"]
        ball_x_center = new_state["ball_x"]
        ball_y = new_state["ball_y"]

        if paddle_tag.endswith("right"):
            ball_x_outer = ball_x_center + self.settings["ball_radius"]
        else:
            ball_x_outer = ball_x_center - self.settings["ball_radius"]

        # Check y-overlap: ball must be within paddle's vertical range
        if ball_y > paddle_y + paddle_height / 2.0 or ball_y < paddle_y - paddle_height / 2.0:
            return False

        # Check x-overlap depending on left or right paddle
        if paddle_tag.endswith("right"):
            return bool(ball_x_center < paddle_x + paddle_width / 2.0
                        and ball_x_outer > paddle_x - paddle_width / 2.0)

        if paddle_tag.endswith("left"):
            return bool(ball_x_center > paddle_x - paddle_width / 2.0
                        and ball_x_outer < paddle_x + paddle_width / 2.0)
        return False

    def _hard_limit(self, v: float, v_min: float, v_max: float) -> float:
        """Clamp a value to [v_min, v_max]."""
        return max(min(v, v_max), v_min)

    def _put_in_boundaries(self, state: dict) -> dict:
        """Clamp all state variables to their valid ranges."""
        for key in state:
            if key == "ball_vel_dir":
                state[key] = self._mod_angle(state[key])
            elif (key + "_min") in self.settings and (key + "_max") in self.settings:
                state[key] = self._hard_limit(state[key], self.settings[key + "_min"], self.settings[key + "_max"])
        return state

    def _update_dynamics(self, prev_state: dict, left_action: int, right_action: int) -> tuple[dict, dict]:
        """Compute the next game state from current state and actions.
        Handles ball movement, wall/paddle collisions, scoring, and resets."""
        new_state = {}
        info = {}

        # Update paddle positions
        new_state["paddle_left_y"] = self._apply_paddle_action(prev_state["paddle_left_y"], left_action)
        new_state["paddle_right_y"] = self._apply_paddle_action(prev_state["paddle_right_y"], right_action)

        # Ball movement: convert (direction, speed) into updated (x, y) position
        vel_y = np.cos(prev_state["ball_vel_dir"]) * prev_state["ball_vel_magn"]
        vel_x = np.sin(prev_state["ball_vel_dir"]) * prev_state["ball_vel_magn"]
        new_state["ball_x"] = prev_state["ball_x"] + vel_x

        # Check for score updates (ball passed paddle)
        point_left, point_right, ball_reset = False, False, False
        if new_state["ball_x"] < self.settings["ball_x_min_point"]:
            point_right = True
            ball_reset = True
        elif new_state["ball_x"] > self.settings["ball_x_max_point"]:
            point_left = True
            ball_reset = True

        # Reset ball and paddles to center after scoring
        if ball_reset:
            new_state["ball_x"] = self.settings["center_point_x"]
            new_state["ball_y"] = self.settings["center_point_y"]
            new_state["ball_vel_dir"] = self.rng.uniform(0.0, 2.0 * np.pi)
            new_state["paddle_left_y"] = self.rng.uniform(self.settings["center_point_y"] * 0.25, self.settings["center_point_y"] * 1.75)
            new_state["paddle_right_y"] = self.rng.uniform(self.settings["center_point_y"] * 0.25, self.settings["center_point_y"] * 1.75)
            info["wall_collision"] = False
            info["paddle_collision"] = None
        else:
            new_state["ball_y"] = prev_state["ball_y"] + vel_y
            new_state["ball_vel_dir"] = prev_state["ball_vel_dir"]

            # Wall collisions (top and bottom boundaries)
            wall_collision = False
            if new_state["ball_y"] > self.settings["ball_y_max"]:
                new_state["ball_y"] = self.settings["ball_y_max"] - (new_state["ball_y"] - self.settings["ball_y_max"])
                new_state["ball_vel_dir"] = self._angle_flip(new_state["ball_vel_dir"], axis="x")
                new_state["ball_vel_dir"] += self.rng.standard_normal() * self.settings["ball_bounce_noise"]
                wall_collision = True
            elif new_state["ball_y"] < self.settings["ball_y_min"]:
                new_state["ball_y"] = self.settings["ball_y_min"] - (new_state["ball_y"] - self.settings["ball_y_min"])
                new_state["ball_vel_dir"] = self._angle_flip(new_state["ball_vel_dir"], axis="x")
                new_state["ball_vel_dir"] += self.rng.standard_normal() * self.settings["ball_bounce_noise"]
                wall_collision = True
            info["wall_collision"] = wall_collision

            # Paddle collisions
            paddle_collision = None
            if self._ball_collision("paddle_left", new_state, prev_state):
                new_state["ball_x"] = ((self.settings["paddle_left_x"] + self.settings["paddle_width"] / 2.0) * 2
                                    - (new_state["ball_x"] - self.settings["ball_radius"] * 2))
                new_state["ball_vel_dir"] = self._angle_flip(new_state["ball_vel_dir"], axis="y")
                new_state["ball_vel_dir"] += self.rng.standard_normal() * self.settings["ball_bounce_noise"]
                paddle_collision = "left"
            elif self._ball_collision("paddle_right", new_state, prev_state):
                new_state["ball_x"] = ((self.settings["paddle_right_x"] - self.settings["paddle_width"] / 2.0) * 2
                                    - (new_state["ball_x"] + self.settings["ball_radius"] * 2))
                new_state["ball_vel_dir"] = self._angle_flip(new_state["ball_vel_dir"], axis="y")
                new_state["ball_vel_dir"] += self.rng.standard_normal() * self.settings["ball_bounce_noise"]
                paddle_collision = "right"
            info["paddle_collision"] = paddle_collision

            # Add small ball position noise
            new_state["ball_x"] += self.rng.standard_normal() * self.settings["ball_pos_noise"]
            new_state["ball_y"] += self.rng.standard_normal() * self.settings["ball_pos_noise"]
            new_state["ball_vel_dir"] += self.rng.standard_normal() * self.settings["ball_vel_dir_noise"]

        # Update scores
        new_state["score_left"] = prev_state["score_left"] + int(point_left)
        new_state["score_right"] = prev_state["score_right"] + int(point_right)

        # Reset scores when a player reaches max points
        if max(new_state["score_right"], new_state["score_left"]) >= self.settings["max_points"]:
            new_state["score_left"] = 0
            new_state["score_right"] = 0

        new_state["ball_vel_magn"] = prev_state["ball_vel_magn"]
        new_state = self._put_in_boundaries(new_state)
        
        info["ball_reset"] = ball_reset

        return new_state, info

    def step(self, left_action: Optional[int] = None, right_action: Optional[int] = None) -> tuple[dict, dict]:
        """Advance the game by one timestep. If an action is None, the AI controller determines it automatically."""
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        # Use AI controller if no action provided
        if left_action is None:
            left_action, self.ai_bias_left = self._get_ai_action(self.state["paddle_left_y"], 
                                                                 self.state["ball_y"], self.ai_bias_left)   
        if right_action is None:
            right_action, self.ai_bias_right = self._get_ai_action(self.state["paddle_right_y"], 
                                                                   self.state["ball_y"], self.ai_bias_right)
        self.t += 1
        self.prev_state = self.state
        new_state, info = self._update_dynamics(self.state, left_action, right_action)
        self.state = new_state

        info["left_action"] = left_action
        info["right_action"] = right_action
        
        return new_state, info
        
