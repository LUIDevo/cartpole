import numpy as np

DT = 1.0 / 60.0
GRAVITY = 980.0

HALF_TRACK = 500.0
MAX_CART_VEL = 1000.0
MAX_POLE_ANGVEL = 10.0
MAX_POLE_ANGLE = np.pi

POLE_LEN = 200.0
POLE_WIDTH = 20.0
R_COM = POLE_LEN / 2.0
I_COM_PER_M = (POLE_WIDTH**2 + POLE_LEN**2) / 12.0
ANGULAR_DAMP = 1.0

STATE_DIM = 8

GOAL_SWITCH_MIN_TICKS = 180
GOAL_SWITCH_MAX_TICKS = 600

W_ANGVEL = 0.003
W_POS = 0.1
W_EDGE = 10.0
UP_BONUS = 0.25
ALIVE_BONUS = 1.0
DEATH_PENALTY = 10.0

# swing-up shaping: pay for rotational speed while an up-goal pole hangs
# below horizontal, capped at the speed sufficient to coast to upright
W_SWING = 0.07
OMEGA_SWING = np.sqrt(4.0 * GRAVITY * R_COM / (I_COM_PER_M + R_COM**2))

NEAR_UP_FRAC = 0.35
CONFIRM_FRAC = 0.3

GOAL_PAIRS = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
GOAL_WEIGHTS = np.array([0.40, 0.25, 0.25, 0.10])


class MathCartPoleVec:
    def __init__(self, n, seed=None, goal_switching=True):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.goal_switching = goal_switching

        z = lambda: np.zeros(n)
        self.x, self.v = z(), z()
        self.theta1, self.omega1 = z(), z()
        self.theta2, self.omega2 = z(), z()
        self.g1 = np.ones(n)
        self.g2 = np.ones(n)
        self.switch_in = np.zeros(n, dtype=np.int64)
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

    def _rand_goal_pairs(self, k):
        pick = self.rng.choice(len(GOAL_PAIRS), size=k, p=GOAL_WEIGHTS)
        return GOAL_PAIRS[pick, 0], GOAL_PAIRS[pick, 1]

    def _rand_angles(self, k):
        wide = self.rng.uniform(-np.pi, np.pi, size=k)
        near_up = self.rng.normal(0.0, 0.3, size=k)
        return np.where(self.rng.random(k) < NEAR_UP_FRAC, near_up, wide)

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

        self.x[idx] = u(-250.0, 250.0)
        self.v[idx] = u(-60.0, 60.0)
        self.theta1[idx] = u(-np.pi, np.pi)
        self.omega1[idx] = u(-2.0, 2.0)
        self.theta2[idx] = u(-np.pi, np.pi)
        self.omega2[idx] = u(-2.0, 2.0)

        k = np.size(idx)
        self.g1[idx], self.g2[idx] = self._rand_goal_pairs(k)
        self.switch_in[idx] = self.rng.integers(
            GOAL_SWITCH_MIN_TICKS, GOAL_SWITCH_MAX_TICKS, size=np.shape(idx))

    def reset_all(self):
        self._randomize(np.arange(self.n))
        return self.observe()

    def reset_at(self, i):
        self._randomize(np.array([i]))
        return self.observe()[i]

    def reset_where(self, mask):
        self._randomize(np.flatnonzero(mask))
        return self.observe()

    def set_goal(self, i, g1, g2):
        self.g1[i] = g1
        self.g2[i] = g2

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

        if self.goal_switching:
            self.switch_in -= 1
            expired = self.switch_in <= 0
            if expired.any():
                k = int(expired.sum())
                self.g1[expired], self.g2[expired] = self._rand_goal_pairs(k)
                self.switch_in[expired] = self.rng.integers(
                    GOAL_SWITCH_MIN_TICKS, GOAL_SWITCH_MAX_TICKS, size=k)

        done = self.terminal()
        rew = self.reward() - DEATH_PENALTY * done
        return self.observe(), rew, done

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
            self.g1,
            self.g2,
        ], axis=1).astype(np.float32)

    def reward(self):
        pos_n = self.x / HALF_TRACK
        c1, c2 = np.cos(self.theta1), np.cos(self.theta2)
        edge = np.maximum(0.0, np.abs(pos_n) - 0.8)
        bonus = (UP_BONUS * ((self.g1 > 0) & (c1 > 0.9))
                 + UP_BONUS * ((self.g2 > 0) & (c2 > 0.9)))
        return (ALIVE_BONUS
                + 0.5 * (self.g1 * c1 + self.g2 * c2)
                + bonus
                - W_ANGVEL * (self.omega1**2 + self.omega2**2)
                - W_POS * pos_n**2
                - W_EDGE * edge**2)

    def terminal(self):
        return np.abs(self.x) >= HALF_TRACK
