"""Piecewise constant-jerk commands shared by the QP and the 1 kHz servo."""
from dataclasses import dataclass

import numpy as np

from mstack.config.policy_control import POLICY_EE_LINEAR_VELOCITY, POLICY_EE_ANGULAR_VELOCITY
from mstack.robots.fr3_kinematics import FR3_Q_MIN, FR3_Q_MAX

# FR3 FCI position-based velocity limits (not Panda/FER limits).
# https://frankarobotics.github.io/docs/robot_specifications.html
HW_MIN = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, .4398, -3.0508])
HW_MAX = np.array([2.9007, 1.8361, 2.9007, -.1169, 2.8763, 4.6216, 3.0508])
OFFSET = np.array([.6599, .2517, .2, .3533, .5757, .4878, .4628])
DECEL = np.array([6., 2.585, 3.5, 4., 17., 5.5, 17.])


@dataclass(frozen=True)
class Settings:
    horizon: int = 30
    knot_dt: float = .01
    servo_dt: float = .001
    velocity: float = 1.5
    acceleration: float = 6.
    jerk: float = 3000.
    ee_linear_velocity: float = POLICY_EE_LINEAR_VELOCITY
    ee_angular_velocity: float = POLICY_EE_ANGULAR_VELOCITY
    position_margin: float = .01
    solver_tolerance: float = 1e-4
    tolerance: float = 2e-7
    tracking_error: float = .15
    planning_period: float = .01
    activation_lead_ticks: int = 20

    def __post_init__(self):
        vals = [self.knot_dt, self.servo_dt, self.velocity, self.acceleration, self.jerk,
                self.position_margin, self.solver_tolerance, self.tolerance, self.tracking_error,
                self.planning_period, self.ee_linear_velocity, self.ee_angular_velocity]
        if (not np.isfinite(vals).all() or min(vals) <= 0 or self.horizon < 2
                or self.servo_dt != .001 or self.velocity > 1.5 or self.acceleration > 6
                or self.jerk > 3000 or self.activation_lead_ticks < 2):
            raise ValueError("invalid FR3 policy-control settings")

    @property
    def lower(self):
        return FR3_Q_MIN + self.position_margin

    @property
    def upper(self):
        return FR3_Q_MAX - self.position_margin

    def velocity_lines(self):
        """Inner piecewise-linear envelope: sign*v + slope*sign*q <= bound.

        Chords lie below the concave sqrt envelope, unlike unsafe tangents.
        The velocity box supplies the flat cap. Two chords per side suffice;
        continuous feasibility is checked on cubic Bezier control points.
        """
        lines = []
        for sign, hardware, configured in ((1, HW_MAX, self.upper), (-1, HW_MIN, self.lower)):
            minimum = sign * (hardware - configured)
            maximum = (self.velocity + OFFSET)**2 / (2*DECEL)
            maximum = np.maximum(maximum, minimum + .001)
            points = np.geomspace(minimum, maximum, 3)
            values = np.sqrt(2*DECEL*points) - OFFSET
            if np.min(values) <= 0:
                raise ValueError("position margin too small for the velocity envelope")
            for k in range(2):
                slope = (values[k+1] - values[k]) / (points[k+1] - points[k])
                intercept = values[k] - slope*points[k]
                bound = slope*sign*hardware + intercept
                lines.append((sign, slope, bound))
        return lines


def bezier(states, dt):
    q0, v0, a0 = states[:-1, 0], states[:-1, 1], states[:-1, 2]
    q1, v1 = states[1:, 0], states[1:, 1]
    q = np.stack([q0, q0+dt*v0/3, q1-dt*v1/3, q1], axis=1)
    v_mid = v0+dt*a0/2
    v = np.stack([v0, v_mid, v1], axis=1)
    v_cubic = np.stack([v0, (v0+2*v_mid)/3, (2*v_mid+v1)/3, v1], axis=1)
    return q, v, v_cubic


@dataclass(frozen=True)
class Plan:
    states: np.ndarray  # (N+1, 3, 7): q/v/a
    dt: float
    elapsed_ms: float = 0.
    solve_ms: float = 0.
    iterations: int = 0

    def __post_init__(self):
        states = np.array(self.states, dtype=float, copy=True)
        states.flags.writeable = False
        object.__setattr__(self, "states", states)

    @property
    def duration(self):
        return self.dt*(len(self.states)-1)

    def sample(self, seconds):
        if not np.isfinite(seconds) or seconds < -1e-10:
            raise ValueError("invalid plan sample time")
        if seconds >= self.duration:
            return self.states[-1].copy()
        k = min(int(max(seconds, 0)/self.dt), len(self.states)-2)
        t = seconds-k*self.dt
        q, v, a = self.states[k]
        j = (self.states[k+1, 2]-a)/self.dt
        return np.array([q+v*t+a*t*t/2+j*t*t*t/6, v+a*t+j*t*t/2, a+j*t])

    @classmethod
    def rest(cls, q, settings):
        states = np.zeros((2, 3, 7))
        states[:, 0] = q
        return cls(states, settings.knot_dt)

    def validate(self, settings, kinematics=None):
        s = settings
        x = self.states
        if (x.ndim != 3 or x.shape[1:] != (3, 7) or len(x) < 2
                or not np.isfinite(x).all() or not np.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("invalid trajectory")
        q, v, vc = bezier(x, self.dt)
        jerk = np.diff(x[:, 2], axis=0)/self.dt
        # Validate integration independently of the solver's status/residual.
        dp = x[:-1, 1]*self.dt + x[:-1, 2]*self.dt**2/2 + jerk*self.dt**3/6
        dv = x[:-1, 2]*self.dt + jerk*self.dt**2/2
        violations = [np.max(s.lower-q), np.max(q-s.upper),
                      np.max(abs(v))-s.velocity, np.max(abs(x[:, 2]))-s.acceleration,
                      np.max(abs(jerk))-s.jerk,
                      np.max(abs(np.diff(x[:, 0], axis=0)-dp)),
                      np.max(abs(np.diff(x[:, 1], axis=0)-dv)),
                      np.max(abs(x[-1, 1:]))]
        for sign, slope, bound in s.velocity_lines():
            violations.append(np.max(sign*(vc+slope*q)-bound))
        violation = float(max(0., *violations))
        if violation > s.tolerance:
            raise ValueError(f"trajectory exceeds continuous constraints by {violation:.3g}")
        from mstack.robots.policy_control.cartesian import validate_speed
        validate_speed(self, s, kinematics)
        return violation
