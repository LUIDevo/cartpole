import numpy as np

DT = 1.0 / 60.0
GRAVITY = 980.0

HALF_TRACK = 500.0
MAX_CART_VEL = 1000.0
MAX_POLE_ANGVEL = 10.0
MAX_POLE_ANGLE = np.pi
FAIL_ANGLE = np.pi / 2.0

POLE_LEN = 200.0
POLE_WIDTH = 20.0
R_COM = POLE_LEN / 2.0
I_PER_M = (POLE_WIDTH**2 + POLE_LEN**2) / 12.0 + R_COM**2
ANGULAR_DAMP = 1.0

STATE_DIM = 4


class MathCartPoleVec:
    def __init__(self, n, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)

        z = lambda: np.zeros(n)
        self.x, self.v = z(), z()
        self.theta, self.omega = z(), z()
        self.cart_mass = z()
        self.pole_mass = z()
        self.max_force = z()
        self.max_power = z()
        self.deadzone = z()
        self.exponent = z()
        self.bias = z()

        self._randomize(np.arange(n))

    def _randomize(self, idx):
        u = lambda lo, hi: self.rng.uniform(lo, hi, size=np.shape(idx))
        self.cart_mass[idx] = u(3.0, 8.0)
        self.pole_mass[idx] = u(1.0, 4.0)
        self.max_force[idx] = u(8000.0, 16000.0)
        self.max_power[idx] = u(4000.0, 7000.0)
        self.deadzone[idx] = u(0.0, 0.05)
        self.exponent[idx] = u(0.9, 1.2)
        self.bias[idx] = u(-0.03, 0.03)

        self.x[idx] = 0.0
        self.v[idx] = u(-60.0, 60.0)
        self.theta[idx] = u(-0.25, 0.25)
        self.omega[idx] = u(-1.0, 1.0)

    def reset_all(self):
        self._randomize(np.arange(self.n))
        return self.observe()

    def reset_at(self, i):
        self._randomize(np.array([i]))
        return self.observe()[i]

    def _motor_force(self, u):
        u = np.clip(u + self.bias, -1.0, 1.0)
        mag = np.abs(u)
        mag = np.where(mag < self.deadzone,
                       0.0,
                       (mag - self.deadzone) / (1.0 - self.deadzone))
        mag = mag ** self.exponent
        return np.sign(u) * mag * self.max_force

    def step(self, commands):
        u = np.clip(np.asarray(commands, dtype=np.float64), -1.0, 1.0)

        total_mass = self.cart_mass + self.pole_mass
        accel = self._motor_force(u) / total_mass
        max_speed = np.minimum(self.max_power / (total_mass * 0.5), MAX_CART_VEL)

        v_old = self.v
        v = np.clip(v_old + accel * DT, -max_speed, max_speed)
        x = self.x + v * DT
        at_lo, at_hi = x <= -HALF_TRACK, x >= HALF_TRACK
        x = np.clip(x, -HALF_TRACK, HALF_TRACK)
        v = np.where((at_lo & (v < 0.0)) | (at_hi & (v > 0.0)), 0.0, v)

        a_pivot = (v - v_old) / DT
        alpha = R_COM * (GRAVITY * np.sin(self.theta)
                         - a_pivot * np.cos(self.theta)) / I_PER_M
        omega = (self.omega + alpha * DT) * max(0.0, 1.0 - ANGULAR_DAMP * DT)
        theta = self.theta + omega * DT

        self.x, self.v, self.theta, self.omega = x, v, theta, omega
        return self.observe(), self.reward(), self.terminal()

    def observe(self):
        wrapped = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        return np.stack([
            self.v / MAX_CART_VEL,
            self.omega / MAX_POLE_ANGVEL,
            wrapped / MAX_POLE_ANGLE,
            np.abs(self.x) / HALF_TRACK,
        ], axis=1).astype(np.float32)

    def reward(self):
        pos_n = self.x / HALF_TRACK
        return 1.0 - (self.theta**2 + 0.1 * self.omega**2 + 0.5 * pos_n**2)

    def terminal(self):
        return (np.abs(self.theta) > FAIL_ANGLE) | (np.abs(self.x) >= HALF_TRACK)
