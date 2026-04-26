import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def body_frame_vector(dx: float, dy: float, theta: float) -> np.ndarray:
    """
    Convert world-frame vector into robot body frame.
    Robot heading theta is measured in world frame.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # Rotation by -theta
    x_body = c * dx + s * dy
    y_body = -s * dx + c * dy
    return np.array([x_body, y_body], dtype=np.float32)


def step_differential_drive(
    state: np.ndarray,
    action: np.ndarray,
    dt: float,
    v_max: float,
    omega_max: float,
    acc_limit: float,
    omega_acc_limit: float,
) -> np.ndarray:
    """
    State: [x, y, theta, v, omega]
    Action: normalized [a_v, a_omega] in [-1, 1]
    """
    x, y, theta, v, omega = state

    a_v = float(np.clip(action[0], -1.0, 1.0))
    a_omega = float(np.clip(action[1], -1.0, 1.0))

    # First version: no reverse. Map [-1, 1] -> [0, v_max]
    v_cmd = 0.5 * (a_v + 1.0) * v_max
    omega_cmd = a_omega * omega_max

    # Acceleration limits
    dv = np.clip(v_cmd - v, -acc_limit * dt, acc_limit * dt)
    domega = np.clip(
        omega_cmd - omega,
        -omega_acc_limit * dt,
        omega_acc_limit * dt,
    )

    v = v + dv
    omega = omega + domega

    x = x + v * np.cos(theta) * dt
    y = y + v * np.sin(theta) * dt
    theta = wrap_angle(theta + omega * dt)

    return np.array([x, y, theta, v, omega], dtype=np.float32)
