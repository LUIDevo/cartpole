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
I_COM_PER_M = (POLE_WIDTH**2 + POLE_LEN**2) / 12.0
ANGULAR_DAMP = 1.0

STATE_DIM = 6


class MathCartPoleVec:
    def __init__(self, n, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)

        z = lambda: np.zeros(n)
        self.x, self.v = z(), z()
        self.theta1, self.omega1 = z(), z()
        self.theta2, self.omega2 = z(), z()
        self.cart_mass = z()
        self.pole_mass_belief = z()
        self.m1 = z()
        self.m2 = z()
        self.max_force = z()
        self.max_power = z()
        self.deadzone = z()
        self.exponent = z()
        self.bias = z()

        self._randomize(np.arange(n))

    def _randomize(self, idx):
        u = lambda lo, hi: self.rng.uniform(lo, hi, size=np.shape(idx))
        self.cart_mass[idx] = u(3.0, 8.0)
        self.pole_mass_belief[idx] = u(1.0, 4.0)
        self.m1[idx] = u(1.0, 5.0)
        self.m2[idx] = u(1.0, 5.0)
        self.max_force[idx] = u(8000.0, 16000.0)
        self.max_power[idx] = u(4000.0, 7000.0)
        self.deadzone[idx] = u(0.0, 0.05)
        self.exponent[idx] = u(0.9, 1.2)
        self.bias[idx] = u(-0.03, 0.03)

        self.x[idx] = 0.0
        self.v[idx] = u(-60.0, 60.0)
        self.theta1[idx] = u(-0.25, 0.25)
        self.omega1[idx] = u(-1.0, 1.0)
        self.theta2[idx] = u(-0.25, 0.25)
        self.omega2[idx] = u(-1.0, 1.0)

    def reset_all(self):
        self._randomize(np.arange(self.n))
        return self.observe()

    def reset_at(self, i):
        self._randomize(np.array([i]))
        return self.observe()[i]

    def reset_where(self, mask):
        self._randomize(np.flatnonzero(mask))
        return self.observe()

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

        total_mass = self.cart_mass + self.pole_mass_belief
        accel = self._motor_force(u) / total_mass
        max_speed = np.minimum(self.max_power / (total_mass * 0.5), MAX_CART_VEL)

        v_old = self.v
        v = np.clip(v_old + accel * DT, -max_speed, max_speed)
        x = self.x + v * DT
        at_lo, at_hi = x <= -HALF_TRACK, x >= HALF_TRACK
        x = np.clip(x, -HALF_TRACK, HALF_TRACK)
        v = np.where((at_lo & (v < 0.0)) | (at_hi & (v > 0.0)), 0.0, v)
        a_pivot = (v - v_old) / DT

        # Double pendulum on a pivot with prescribed acceleration (Lagrange).
        # theta measured from upright, positive tips toward +x; the kinematic
        # cart takes no back-reaction, but the rods couple to each other.
        m1, m2 = self.m1, self.m2
        L, r = POLE_LEN, R_COM
        s1, c1 = np.sin(self.theta1), np.cos(self.theta1)
        s2, c2 = np.sin(self.theta2), np.cos(self.theta2)
        delta = self.theta1 - self.theta2
        sd, cd = np.sin(delta), np.cos(delta)

        m11 = I_COM_PER_M * m1 + m1 * r * r + m2 * L * L
        m22 = I_COM_PER_M * m2 + m2 * r * r
        a_c = m2 * L * r
        m12 = a_c * cd
        h = m1 * r + m2 * L

        rhs1 = h * (GRAVITY * s1 - a_pivot * c1) - a_c * sd * self.omega2**2
        rhs2 = m2 * r * (GRAVITY * s2 - a_pivot * c2) + a_c * sd * self.omega1**2

        det = m11 * m22 - m12 * m12
        alpha1 = (rhs1 * m22 - m12 * rhs2) / det
        alpha2 = (rhs2 * m11 - m12 * rhs1) / det

        damp = max(0.0, 1.0 - ANGULAR_DAMP * DT)
        omega1 = (self.omega1 + alpha1 * DT) * damp
        omega2 = (self.omega2 + alpha2 * DT) * damp
        theta1 = self.theta1 + omega1 * DT
        theta2 = self.theta2 + omega2 * DT

        self.x, self.v = x, v
        self.theta1, self.omega1 = theta1, omega1
        self.theta2, self.omega2 = theta2, omega2
        return self.observe(), self.reward(), self.terminal()

    def observe(self):
        wrap1 = np.arctan2(np.sin(self.theta1), np.cos(self.theta1))
        wrap2 = np.arctan2(np.sin(self.theta2), np.cos(self.theta2))
        return np.stack([
            self.v / MAX_CART_VEL,
            self.omega1 / MAX_POLE_ANGVEL,
            wrap1 / MAX_POLE_ANGLE,
            self.x / HALF_TRACK,
            self.omega2 / MAX_POLE_ANGVEL,
            wrap2 / MAX_POLE_ANGLE,
        ], axis=1).astype(np.float32)

    def reward(self):
        pos_n = self.x / HALF_TRACK
        return 1.0 - (self.theta1**2 + self.theta2**2
                      + 0.1 * (self.omega1**2 + self.omega2**2)
                      + 0.5 * pos_n**2)

    def terminal(self):
        return ((np.abs(self.theta1) > FAIL_ANGLE)
                | (np.abs(self.theta2) > FAIL_ANGLE)
                | (np.abs(self.x) >= HALF_TRACK))
