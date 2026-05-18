# RingWorldEnv: a minimal, gym-like environment for testing edge-reward behaviors
# Author: ChatGPT
# Requirements: gymnasium (>=0.28) or adapt the return signatures for classic gym
#   pip install gymnasium pillow numpy
#
# Key idea: The agent moves on a ring with N discrete positions. The reward is attached
# to the *transition* (edge): moving clockwise yields +1, moving counter-clockwise yields -1
# (configurable). This makes it impossible to express purely with node (state) rewards and
# directly tests policies that must prefer a direction despite revisiting the same states.

from __future__ import annotations
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback to classic gym if needed
    import gym
    from gym import spaces

from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


class RingWorldEnv(gym.Env):
    """
    A tiny ring world with edge-based rewards.

    Observation modes:
      - 'index': single integer state in [0, n-1] (spaces.Discrete)
      - 'one_hot': one-hot vector of length n (spaces.Box)
      - 'angle': (cos(theta), sin(theta)) (spaces.Box)  <-- good for function approximation

    Actions:
      - 0: stay
      - 1: move clockwise (CW)
      - 2: move counter-clockwise (CCW)

    Rewards (edge-based by default):
      - reward_cw for CW moves, reward_ccw for CCW moves
      - Optional slip probability can invert the chosen action to simulate stochastic transitions

    Episode control:
      - terminated is always False (no terminal state on a ring)
      - truncated becomes True when step_count reaches max_steps

    Rendering:
      - render_mode can be: "human" (prints ascii), "rgb_array" (returns HxWx3 uint8)
      - The image draws the ring and a marker at the agent's current position.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n: int = 12,
        obs_mode: str = "angle",   # "index" | "one_hot" | "angle"
        start: Optional[int] = None,   # None => random start
        reward_cw: float = 1.0,
        reward_ccw: float = -1.0,
        slip_prob: float = 0.0,     # probability of flipping the chosen action
        max_steps: int = 200,
        seed: Optional[int] = None,
        # --- rendering ---
        render_mode: Optional[str] = None,  # "human" | "rgb_array" | None
        img_size: int = 256,
        ring_radius_ratio: float = 0.38,    # fraction of img_size
        ring_thickness: int = 6,
        action_mode: str = "continuous",
        deadzone: float = 0.33,
    ) -> None:
        super().__init__()
        assert n >= 3, "n must be >= 3"
        assert obs_mode in {"index", "one_hot", "angle"}
        assert 0.0 <= slip_prob < 1.0
        assert render_mode in {None, "human", "rgb_array"}
        assert 0.1 <= ring_radius_ratio <= 0.48
        assert action_mode in {"continuous", "discrete"}
        
        self.action_mode = action_mode
        self.n = n
        self.obs_mode = obs_mode
        self.start = start
        self.reward_cw = reward_cw
        self.reward_ccw = reward_ccw
        self.slip_prob = slip_prob
        self.max_steps = int(max_steps)
        self.deadzone = float(deadzone)


        # Rendering config
        self.render_mode = render_mode
        self.img_size = int(img_size)
        self.ring_radius_ratio = float(ring_radius_ratio)
        self.ring_thickness = int(ring_thickness)

        # RNG
        self._rng = np.random.default_rng(seed)


        # Spaces
        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        if obs_mode == "index":
            self.observation_space = spaces.Box(low=0.0, high=float(n-1), shape=(1,), dtype=np.float32)
        elif obs_mode == "one_hot":
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        else:  # angle: cos, sin
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # State
        self.state: int = 0
        self.step_count: int = 0

        # Cache last frame for human mode if needed
        self._last_frame: Optional[np.ndarray] = None

        self.horizon = self.max_steps


    # --- Utility helpers ---
    def _state_to_obs(self, s: int):
        if self.obs_mode == "index":
            return np.array([float(s)], dtype=np.float32)
        elif self.obs_mode == "one_hot":
            x = np.zeros(self.n, dtype=np.float32)
            x[s] = 1.0
            return x
        else:  # angle
            theta = 2.0 * np.pi * (s / self.n)
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) 

    def _move(self, s: int, a: int) -> Tuple[int, int]:
        """
        Apply action with optional slip. Returns (action_used, next_state)
        action_used is 1 or 2 after slip resolution.
        """
        a_used = a
        if self.slip_prob > 0.0 and self._rng.random() < self.slip_prob and a_used:
            a_used = 3 - a  # flip action
        if a_used == 1:  # CW
            ns = (s + 1) % self.n
        elif a_used == 2: # CCW
            ns = (s - 1) % self.n
        else:  # Stay
            ns = s
        return a_used, ns

    # --- Gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # If user passes a seed at reset-time, recreate RNG for determinism
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        if self.start is None:
            self.state = int(self._rng.integers(0, self.n))
        else:
            self.state = int(self.start % self.n)
        obs = self._state_to_obs(self.state)
        # For gymnasium, if render_mode is rgb_array, you may return a first frame via render()
        if self.render_mode == "rgb_array":
            self._last_frame = self._draw_frame()
        return obs

    def step(self, action: int):
        if self.action_mode == "discrete":
            assert self.action_space.contains(action), "invalid action"
            a_bin = int(action)
        else:
            a_val = float(np.asarray(action).reshape(-1)[0])
            if abs(a_val) <= self.deadzone:
                a_bin = 0
            elif a_val > 0:
                a_bin = 1
            else:
                a_bin = 2

        s = self.state
        a_used, ns = self._move(s, a_bin)

        # Edge-based reward: depends on the direction actually executed
        reward = self.reward_cw if a_used == 1 else self.reward_ccw if a_used == 2 else 0.0

        self.state = ns
        self.step_count += 1

        terminated = False             # ring has no terminal states
        truncated = self.step_count >= self.max_steps
        obs = self._state_to_obs(self.state)
        info = {"action_used": a_used}

        # draw frame if needed
        if self.render_mode == "rgb_array":
            self._last_frame = self._draw_frame()

        return obs, float(reward), terminated, truncated, {"action_used": a_used}

    # --- Rendering ---
    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode
        if self.render_mode == "human":
            ring = ["o"] * self.n
            ring[self.state] = "*"
            print(" ".join(ring))
            return None
        elif self.render_mode == "rgb_array":
            if self._last_frame is None:
                self._last_frame = self._draw_frame()
            return self._last_frame
        else:
            # If no render_mode was set, default to returning an ANSI string for convenience
            ring = ["o"] * self.n
            ring[self.state] = "*"
            return " ".join(ring)

    def close(self):
        pass

    # --- Frame drawing ---
    def _draw_frame(self) -> np.ndarray:
        """Return an HxWx3 uint8 image representing the ring and agent marker."""
        H = W = self.img_size
        # Background
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Ring geometry
        cx, cy = W // 2, H // 2
        R = int(self.ring_radius_ratio * min(W, H))
        t = self.ring_thickness
        # Outer and inner circles (to get thickness)
        bbox_outer = [cx - R - t//2, cy - R - t//2, cx + R + t//2, cy + R + t//2]
        bbox_inner = [cx - R + t//2, cy - R + t//2, cx + R - t//2, cy + R - t//2]
        draw.ellipse(bbox_outer, outline=(180, 180, 200), width=t)
        draw.ellipse(bbox_inner, outline=None)

        # Ticks (optional)
        for k in range(self.n):
            ang = 2.0 * np.pi * (k / self.n)
            x0 = cx + int((R - t*1.5) * np.cos(ang))
            y0 = cy + int((R - t*1.5) * np.sin(ang))
            x1 = cx + int((R + t*1.5) * np.cos(ang))
            y1 = cy + int((R + t*1.5) * np.sin(ang))
            draw.line((x0, y0, x1, y1), fill=(110, 110, 130), width=2)

        # Agent marker at current state
        theta = 2.0 * np.pi * (self.state / self.n)
        px = cx + int(R * np.cos(theta))
        py = cy + int(R * np.sin(theta))
        r_agent = max(10, t)  # radius of the marker
        draw.ellipse((px - r_agent, py - r_agent, px + r_agent, py + r_agent), fill=(255, 180, 60))

        return np.array(img, dtype=np.uint8)

    def save_env_overview(self, path="ringworld_env.png"):
        """Save the current RingWorld environment as an image."""
        old_mode = self.render_mode
        self.render_mode = "rgb_array"

        frame = self._draw_frame()
        img = Image.fromarray(frame)

        img.save(path)

        self.render_mode = old_mode
        return path

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    # rgb_array render demo
    env = RingWorldEnv(
        n=12,
        obs_mode="angle",
        reward_cw=+1.0,
        reward_ccw=-1.0,
        slip_prob=0.05,
        max_steps=20,
        render_mode="rgb_array",
        img_size=256
    )
    env.save_env_overview("ringworld_env.png")
    obs, info = env.reset(seed=42)
    print("reset:", obs, info)
    total = 0.0
    for t in range(10):
        action = 1  # always try to go CW
        obs, r, term, trunc, info = env.step(action)
        frame = env.render()  # HxWx3 np.uint8
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        total += r
        print(f"t={t:02d} action={action} used={info['action_used']} reward={r:+.1f} obs={obs}")
    print("return=", total)
