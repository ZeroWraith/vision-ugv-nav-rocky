import numpy as np


def simulate_trajectory(n_frames, speed=0.5, turn_freq=0.02,
                        amplitude=8.0, start_angle=0.0):
    """Generate a realistic curved trajectory for demo purposes.

    Produces a path that weaves left/right (sinusoidal turns) with
    varying speed, simulating a UGV navigating through terrain.

    Parameters
    ----------
    n_frames : int
        Number of frames (poses to generate).
    speed : float
        Base forward speed in metres per frame-step.
    turn_freq : float
        Frequency of sinusoidal turns (lower = wider turns).
    amplitude : float
        Lateral amplitude of the weaving motion in metres.
    start_angle : float
        Initial heading in radians.

    Returns
    -------
    poses : (N, 3) array  — [x, y, yaw] in world frame.
    """
    t = np.arange(n_frames, dtype=np.float64)

    # Yaw: base heading + sinusoidal weaving
    yaw = start_angle + amplitude * np.sin(turn_freq * t * 2 * np.pi)

    # Forward velocity: slows during turns
    turn_rate = np.abs(amplitude * turn_freq * 2 * np.pi *
                        np.cos(turn_freq * t * 2 * np.pi))
    v = speed * np.clip(1.0 - turn_rate * 0.3, 0.4, 1.0)

    # Integrate position from velocity + yaw
    dx = v * np.sin(yaw)
    dy = v * np.cos(yaw)
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    poses = np.column_stack([x, y, yaw])
    return poses


# Pre-defined trajectories for each demo scene
SCENE_TRAJECTORIES = {
    "scene_03": dict(speed=0.5, turn_freq=0.015, amplitude=10.0, start_angle=0.0),
    "village":  dict(speed=0.6, turn_freq=0.025, amplitude=6.0,  start_angle=0.3),
    "trail_7":  dict(speed=0.4, turn_freq=0.03,  amplitude=12.0, start_angle=-0.2),
}
