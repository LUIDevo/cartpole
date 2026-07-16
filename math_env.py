import numpy as np

DT = 1.0 / 60.0
GRAVITY = 2000.0

HALF_TRACK = 500.0
MAX_CART_VEL = 1000.0
MAX_POLE_ANGVEL = 10.0
MAX_POLE_ANGLE = np.pi

POLE_LEN = 200.0
POLE_WIDTH = 20.0
R_COM = POLE_LEN / 2.0
I_COM_PER_M = (POLE_WIDTH**2 + POLE_LEN**2) / 12.0
ANGULAR_DAMP = 0.15
# integration safety: bound pivot impulse (wall stops) and angular velocity
# so the explicit-Euler quadratic Coriolis terms cannot blow up
A_PIVOT_MAX = 20.0 * 2000.0
OMEGA_MAX = 25.0

STATE_DIM = 8

GOAL_SWITCH_MIN_TICKS = 180
GOAL_SWITCH_MAX_TICKS = 600

W_ANGVEL = 0.001
W_POS = 0.1
W_EDGE = 10.0
UP_BONUS = 0.25
ALIVE_BONUS = 1.0
DEATH_PENALTY = 10.0

# swing-up shaping: pay for holding the true system mechanical energy
# (cart-frame, full double-pendulum mass matrix) near the rest energy of
# the goal configuration. Full bonus at E = E_goal (which includes
# standing at the goal), fading linearly over one full energy range.
W_ENERGY = 0.3
# joint bonus: catching *both* poles upright pays distinctly more than
# holding one up, so the up-up catch is worth hunting for
UPUP_BONUS = 0.5

NEAR_UP_FRAC = 0.5
CONFIRM_FRAC = 0.3
# fraction of resets that start inside the up-up capture basin (both
# links near upright, low spin): the wide distribution almost never
# produces recoverable upright states, so without this the balance
# skill is starved of training data
BALANCE_FRAC = 0.25
BALANCE_ANGLE_STD = 0.08
BALANCE_OMEGA_STD = 0.3

# swing->balance handoff (two-specialist deployment): in handoff mode an
# episode ends with a bonus the moment both poles are catchable by the
# balance specialist, so "deliver a catchable state" is the swing
# policy's whole objective. The per-step time cost makes loitering
# strictly unprofitable (minimum-time formulation): without it the
# policy farms the alive/energy shaping forever and never cashes in.
HANDOFF_COS = 0.96
HANDOFF_OMEGA = 1.0
HANDOFF_BONUS = 300.0
# In handoff mode walls are NOT terminal: episode death lets the policy
# reroll awkward starts by driving into a wall (death -10 beats bleeding
# -2/step), and punishing walls harder (-300) makes it cower mid-track
# instead of swinging. With walls as plain end-stops the only way to
# stop the time-cost bleed is a genuine swing-up.
HANDOFF_TIME_COST = 2.0
# reverse curriculum from successful trajectories: when an episode ends
# in handoff, the states 10..180 ticks before the catch (the approach
# corridor) are archived, and half of later resets respawn from a
# perturbed archived state. Gaussian difficulty dials around upright do
# NOT work here: without balance skill, low-spin near-top starts decay
# irrecoverably, so "easy" must mean "on a trajectory that flows into
# the basin" — which archived corridors are by construction.
SEED_FRAC = 0.5
SEED_MAX = 20000
SEED_OFFSETS = np.arange(10, 181, 10)
SEED_NOISE_ANG = 0.05
SEED_NOISE_OMG = 0.15
HIST_LEN = 181

GOAL_PAIRS = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
GOAL_WEIGHTS = np.array([0.40, 0.25, 0.25, 0.10])


class MathCartPoleVec:
    def __init__(self, n, seed=None, goal_switching=True, fixed_goal=None,
                 balance_frac=BALANCE_FRAC, handoff=False):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.fixed_goal = fixed_goal
        self.goal_switching = goal_switching and fixed_goal is None
        self.balance_frac = balance_frac
        self.handoff = handoff
        if handoff:
            self._hist = np.zeros((HIST_LEN, n, 6))
            self._hist_ptr = 0
            self._age = np.zeros(n, dtype=np.int64)
            self._seeds = []
            self.max_offset_seen = 0

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
        self.max_force[idx] = u(16000.0, 32000.0)
        self.max_power[idx] = u(5000.0, 9000.0)
        self.deadzone[idx] = u(0.0, 0.05)
        self.exponent[idx] = u(0.9, 1.2)
        self.bias[idx] = u(-0.03, 0.03)

        k = np.size(idx)
        self.x[idx] = u(-250.0, 250.0)
        self.v[idx] = u(-60.0, 60.0)
        bal = self.rng.random(k) < self.balance_frac
        th = lambda: np.where(bal, self.rng.normal(0.0, BALANCE_ANGLE_STD, k),
                              self._rand_angles(k))
        om = lambda: np.where(bal, self.rng.normal(0.0, BALANCE_OMEGA_STD, k),
                              self.rng.uniform(-2.0, 2.0, k))
        self.theta1[idx] = th()
        self.omega1[idx] = om()
        self.theta2[idx] = th()
        self.omega2[idx] = om()
        if self.handoff:
            self._age[idx] = 0
            # respawn part of the batch from archived approach corridors
            if self._seeds:
                flat = np.atleast_1d(idx)
                use = self.rng.random(k) < SEED_FRAC
                for j in np.flatnonzero(np.atleast_1d(use)):
                    s = self._seeds[self.rng.integers(len(self._seeds))]
                    i = flat[j]
                    self.theta1[i] = s[0] + self.rng.normal(0, SEED_NOISE_ANG)
                    self.omega1[i] = s[1] + self.rng.normal(0, SEED_NOISE_OMG)
                    self.theta2[i] = s[2] + self.rng.normal(0, SEED_NOISE_ANG)
                    self.omega2[i] = s[3] + self.rng.normal(0, SEED_NOISE_OMG)
                    self.x[i] = np.clip(s[4], -0.9 * HALF_TRACK,
                                        0.9 * HALF_TRACK)
                    self.v[i] = s[5]

        if self.fixed_goal is not None:
            self.g1[idx], self.g2[idx] = self.fixed_goal
        else:
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
        a_pivot = np.clip((v - v_old) / DT, -A_PIVOT_MAX, A_PIVOT_MAX)

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
        omega1 = np.clip((self.omega1 + alpha1 * DT) * damp, -OMEGA_MAX, OMEGA_MAX)
        omega2 = np.clip((self.omega2 + alpha2 * DT) * damp, -OMEGA_MAX, OMEGA_MAX)
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
                g1n, g2n = self._rand_goal_pairs(k)
                # curriculum: sometimes the new goal confirms the current
                # configuration, so "keep the pole where it is" (including
                # upright) is well represented in the data
                confirm = self.rng.random(k) < CONFIRM_FRAC
                cur1 = np.where(np.cos(self.theta1[expired]) >= 0.0, 1.0, -1.0)
                cur2 = np.where(np.cos(self.theta2[expired]) >= 0.0, 1.0, -1.0)
                self.g1[expired] = np.where(confirm, cur1, g1n)
                self.g2[expired] = np.where(confirm, cur2, g2n)
                self.switch_in[expired] = self.rng.integers(
                    GOAL_SWITCH_MIN_TICKS, GOAL_SWITCH_MAX_TICKS, size=k)

        # belt-and-braces: any env whose state went non-finite is killed
        bad = ~(np.isfinite(self.theta1) & np.isfinite(self.omega1)
                & np.isfinite(self.theta2) & np.isfinite(self.omega2)
                & np.isfinite(self.v) & np.isfinite(self.x))
        if bad.any():
            self._randomize(np.flatnonzero(bad))

        if self.handoff:
            caught = ((self.g1 > 0) & (self.g2 > 0)
                      & (np.cos(self.theta1) > HANDOFF_COS)
                      & (np.cos(self.theta2) > HANDOFF_COS)
                      & (np.abs(self.omega1) < HANDOFF_OMEGA)
                      & (np.abs(self.omega2) < HANDOFF_OMEGA))
            done = bad | caught
            # archive approach corridors of successful episodes
            self._hist[self._hist_ptr] = np.stack(
                [self.theta1, self.omega1, self.theta2, self.omega2,
                 self.x, self.v], axis=1)
            self._age += 1
            for i in np.flatnonzero(caught & ~bad):
                for o in SEED_OFFSETS[SEED_OFFSETS < self._age[i]]:
                    self._seeds.append(
                        self._hist[(self._hist_ptr - o) % HIST_LEN, i].copy())
                    self.max_offset_seen = max(self.max_offset_seen, int(o))
            if len(self._seeds) > SEED_MAX:
                del self._seeds[:len(self._seeds) - SEED_MAX]
            self._hist_ptr = (self._hist_ptr + 1) % HIST_LEN
            rew = np.where(bad, -DEATH_PENALTY,
                           self.reward() - HANDOFF_TIME_COST
                           + HANDOFF_BONUS * (caught & ~bad))
            return self.observe(), rew, done
        done = self.terminal() | bad
        rew = np.where(bad, -DEATH_PENALTY, self.reward() - DEATH_PENALTY * done)
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
                 + UP_BONUS * ((self.g2 > 0) & (c2 > 0.9))
                 + UPUP_BONUS * ((self.g1 > 0) & (self.g2 > 0)
                                 & (c1 > 0.9) & (c2 > 0.9)))
        m1, m2 = self.m1, self.m2
        L, r = POLE_LEN, R_COM
        m11 = I_COM_PER_M * m1 + m1 * r * r + m2 * L * L
        m22 = I_COM_PER_M * m2 + m2 * r * r
        m12 = m2 * L * r * np.cos(self.theta1 - self.theta2)
        ke = 0.5 * (m11 * self.omega1**2
                    + 2.0 * m12 * self.omega1 * self.omega2
                    + m22 * self.omega2**2)
        pe = GRAVITY * (m1 * r * c1 + m2 * (L * c1 + r * c2))
        e_up = GRAVITY * (m1 * r + m2 * (L + r))
        e_goal = GRAVITY * (m1 * r * self.g1 + m2 * (L * self.g1 + r * self.g2))
        swing = W_ENERGY * np.maximum(
            0.0, 1.0 - np.abs(ke + pe - e_goal) / (2.0 * e_up))
        return (ALIVE_BONUS
                + 0.5 * (self.g1 * c1 + self.g2 * c2)
                + bonus + swing
                - W_ANGVEL * (self.omega1**2 + self.omega2**2)
                - W_POS * pos_n**2
                - W_EDGE * edge**2)

    def terminal(self):
        return np.abs(self.x) >= HALF_TRACK
