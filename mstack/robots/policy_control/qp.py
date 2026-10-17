"""Sparse finite-horizon FR3 QP using the C++ PIQP solver.

Joint actions track all seven joint targets. EE actions use Pinocchio SE(3)
linearization and a soft posture cost; neither mode inverts a Jacobian.
Dynamics, q/v/a/jerk bounds and terminal rest are hard constraints. EE residuals
are soft, so unreachable poses/singularities do not override those bounds.
"""
import time

import numpy as np
import piqp
from scipy import sparse

from mstack.comm.policy_protocol import JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.policy_control.trajectory import Plan, Settings
from mstack.robots.policy_control.cartesian import DIRECTIONS, POLYTOPE_SCALE, CartesianSpeedError
from mstack.robots.policy_control.kinematics import FR3Kinematics


class TrackerQP:
    def __init__(self, kind=JOINT_ABSOLUTE, settings=Settings()):
        if kind not in (JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT):
            raise ValueError("unsupported QP objective")
        self.kind, self.s = kind, settings
        # Joint-output models also need the physical TCP velocity constraints.
        self.kin = FR3Kinematics()
        n, h = settings.horizon, settings.knot_dt
        size = n*28
        self.size = size
        # Each stage contains endpoint q, v, a, and acceleration change d=h*j.
        def ix(k, field, j):
            return k*28+field*7+j
        self.q_indices = np.array([[ix(k, 0, j) for j in range(7)] for k in range(n)])
        A, B = np.zeros((n*21+14, size)), np.zeros((n*21+14, 21))
        for k in range(n):
            for j in range(7):
                for field, coeffs, dcoeff in ((0, (1, h, h*h/2), h*h/6),
                                               (1, (0, 1, h), h/2),
                                               (2, (0, 0, 1), 1)):
                    row = k*21+field*7+j
                    A[row, ix(k, field, j)] = 1
                    A[row, ix(k, 3, j)] = -dcoeff
                    for prevfield, c in enumerate(coeffs):
                        if k:
                            A[row, ix(k-1, prevfield, j)] -= c
                        else:
                            B[row, prevfield*7+j] = c
        for field in (1, 2):
            for j in range(7):
                A[n*21+(field-1)*7+j, ix(n-1, field, j)] = 1
        self.A, self.B = sparse.csc_matrix(A), B
        rows, frees, lower, upper = [], [], [], []
        def variable(k, field, j):
            row, free = np.zeros(size), np.zeros(21)
            if k < 0:
                free[field*7+j] = 1
            else:
                row[ix(k, field, j)] = 1
            return np.r_[row, free]
        def add(expr, lo, hi):
            rows.append(expr[:size]); frees.append(expr[size:]); lower.append(lo); upper.append(hi)
        lines = settings.velocity_lines()
        for k in range(n):
            for j in range(7):
                q0, v0, a0 = [variable(k-1, f, j) for f in range(3)]
                q1, v1 = [variable(k, f, j) for f in range(2)]
                qb = [q0, q0+h*v0/3, q1-h*v1/3, q1]
                vm = v0+h*a0/2
                vb = [v0, vm, v1]
                vc = [v0, (v0+2*vm)/3, (2*vm+v1)/3, v1]
                for expr in qb:
                    add(expr, settings.lower[j], settings.upper[j])
                for expr in vb:
                    add(expr, -settings.velocity, settings.velocity)
                for sign, slope, bound in lines:
                    for q, v in zip(qb, vc):
                        add(sign*(v+slope[j]*q), -np.inf, bound[j])
        full = np.array(rows)
        self.variable = np.any(full != 0, axis=1)
        self.G = sparse.csc_matrix(full[self.variable])
        self.F = sparse.csr_matrix(np.array(frees))
        self.lo, self.hi = np.array(lower), np.array(upper)
        self._cartesian_pattern(n, h, size)
        self.xlo = np.tile(np.r_[settings.lower, np.full(7, -settings.velocity),
                                     np.full(7, -settings.acceleration), np.full(7, -settings.jerk*h)], n)
        self.xhi = np.tile(np.r_[settings.upper, np.full(7, settings.velocity),
                                     np.full(7, settings.acceleration), np.full(7, settings.jerk*h)], n)
        # Keep a fixed Hessian sparsity pattern, including EE q-q off-diagonals.
        self.diag = np.tile(np.r_[np.full(7, 1e-6), np.full(7, 2e-5),
                                  np.full(7, 2e-4), np.full(7, 2e-7/h**2)], n)
        P = sparse.diags(self.diag, format="lil")
        for indices in self.q_indices:
            for i in indices:
                for j in indices:
                    P[i, j] = 1 if i == j else 1e-30
        self.P = sparse.triu(P.tocsc(), format="csc")
        lookup = {(self.P.indices[p], c): p for c in range(size)
                  for p in range(self.P.indptr[c], self.P.indptr[c+1])}
        self.q_entries = [[(i, j, lookup[(indices[i], indices[j])])
                           for j in range(7) for i in range(j+1)] for indices in self.q_indices]
        self.solver = piqp.SparseSolver()
        # Solver stopping criteria are separate from physical trajectory checks.
        self.solver.settings.eps_abs = settings.solver_tolerance
        self.solver.settings.eps_rel = 0.
        self.solver.settings.eps_duality_gap_abs = settings.solver_tolerance
        self.solver.settings.eps_duality_gap_rel = 0.
        self.solver.settings.max_iter = 60
        self.solver.settings.preconditioner_reuse_on_update = True
        self.solver.setup(self.P, np.zeros(size), A=self.A, b=np.zeros(len(A)), G=self.G,
                          h_l=np.full(self.G.shape[0], -1.), h_u=np.full(self.G.shape[0], 1.),
                          x_l=self.xlo, x_u=self.xhi)

    def _cartesian_pattern(self, n, h, size):
        """Fixed sparse rows for transformed quadratic velocity Bezier handles."""
        row, col, stage, direction, joint, coefficient = [], [], [], [], [], []
        kinds = []
        count = 0
        for k in range(n):
            # An endpoint is shared with the next segment. Avoid adding its
            # rows twice with almost identical Jacobians; nonlinear validation
            # still certifies every whole segment, including the initial state.
            for point in (1, 2):
                if k == 0 and point == 1:
                    continue
                terms = [(k*28+7, 1.)] if point == 2 else [((k-1)*28+7, 1.)]
                if point == 1:
                    terms.append(((k-1)*28+14, h/2))
                for d in range(2*len(DIRECTIONS)):
                    kinds.append(d//len(DIRECTIONS))
                    for offset, scale in terms:
                        for j in range(7):
                            row.append(count); col.append(offset+j)
                            stage.append(k); direction.append(d); joint.append(j); coefficient.append(scale)
                    count += 1
        start = self.G.shape[0]
        extra = sparse.csc_matrix((np.ones(len(row)), (row, col)), shape=(count, size))
        self.G = sparse.vstack([self.G, extra], format='csc')
        lookup = {(self.G.indices[p], c): p for c in range(size)
                  for p in range(self.G.indptr[c], self.G.indptr[c+1])}
        self.cart_entries = np.array([lookup[(start+r, c)] for r, c in zip(row, col)])
        self.cart_indices = tuple(np.asarray(x) for x in (stage, direction, joint))
        self.cart_coefficients = np.asarray(coefficient)
        self.cart_kinds = np.asarray(kinds)

    def _cartesian_update(self, state, nominal):
        middle = (np.vstack([state[0], nominal[:-1]])+nominal)/2
        jacobians = np.array([self.kin.velocity_jacobian(q) for q in middle])
        transformed = np.concatenate([DIRECTIONS @ jacobians[:, :3],
                                      DIRECTIONS @ jacobians[:, 3:]], axis=1)
        self.G.data[self.cart_entries] = transformed[self.cart_indices]*self.cart_coefficients
        # Leave headroom for changing Jacobians and the between-sample speed
        # certificate. This factor is not itself a nonlinear safety guarantee.
        limits = np.array([self.s.ee_linear_velocity, self.s.ee_angular_velocity])
        return .95*POLYTOPE_SCALE*limits[self.cart_kinds]

    def plan(self, state, reference, nominal=None, posture=None):
        begin = time.perf_counter()
        solve_ms = iterations = 0
        # A changed posture can invalidate a frozen-Jacobian speed surrogate.
        # Allow one relinearization; reject on a second failure. The installed
        # stopping trajectory remains untouched throughout both attempts.
        for attempt in range(2):
            plan = self._solve(state, reference, nominal, posture)
            solve_ms += plan.solve_ms
            iterations += plan.iterations
            try:
                plan.validate(self.s, self.kin)
            except CartesianSpeedError:
                if attempt:
                    raise
                nominal = plan.states[1:, 0]
            else:
                return Plan(plan.states, plan.dt, (time.perf_counter()-begin)*1000,
                            solve_ms, iterations)

    def _solve(self, state, reference, nominal=None, posture=None):
        begin = time.perf_counter()
        s, n, h = self.s, self.s.horizon, self.s.knot_dt
        state = np.asarray(state, dtype=float)
        if state.shape != (3, 7) or not np.isfinite(state).all() or reference.kind != self.kind:
            raise ValueError("invalid initial state or reference kind")
        if (np.any(state[0] < s.lower-s.tolerance) or np.any(state[0] > s.upper+s.tolerance)
                or np.max(abs(state[1])) > s.velocity+s.tolerance
                or np.max(abs(state[2])) > s.acceleration+s.tolerance):
            raise ValueError("initial state outside limits")
        expected = (n, 7) if self.kind == JOINT_ABSOLUTE else (n, 4, 4)
        if reference.targets.shape != expected or not np.isfinite(reference.targets).all():
            raise ValueError(f"reference must be finite with shape {expected}")
        free = self.F @ state.ravel()
        lo, hi = self.lo-free, self.hi-free
        constant = ~self.variable
        if np.any(lo[constant] > s.tolerance) or np.any(hi[constant] < -s.tolerance):
            raise ValueError("initial Bezier handles infeasible")
        c = np.zeros(self.size)
        nominal = np.tile(state[0], (n, 1)) if nominal is None else np.asarray(nominal)
        posture = state[0] if posture is None else np.asarray(posture)
        if nominal.shape != (n, 7) or posture.shape != (7,) or not np.isfinite(nominal).all() or not np.isfinite(posture).all():
            raise ValueError("invalid nominal trajectory/posture")
        cart_bounds = self._cartesian_update(state, nominal)
        for k, indices in enumerate(self.q_indices):
            if self.kind == JOINT_ABSOLUTE:
                H = 2*np.eye(7)
                c[indices] = -2*reference.targets[k]
            else:
                e, J = self.kin.residual_jacobian(nominal[k], reference.targets[k])
                weights = np.array([10., 10., 10., 1., 1., 1.])
                J, e = weights[:, None]*J, weights*e
                # Damping is a cost, never an inverse or a relaxed safety bound.
                H = 2*(J.T@J + .002*np.eye(7))
                c[indices] = 2*(J.T@(e-J@nominal[k]) - .002*posture)
            for i, j, entry in self.q_entries[k]:
                self.P.data[entry] = H[i, j] + (1e-6 if i == j else 0.)
        xlo, xhi = self.xlo.copy(), self.xhi.copy()
        if self.kind == EE_RELATIVE_WAYPOINT:
            xlo[self.q_indices] = np.maximum(xlo[self.q_indices], nominal-.35)
            xhi[self.q_indices] = np.minimum(xhi[self.q_indices], nominal+.35)
        self.solver.update(P=self.P, c=c, G=self.G, b=self.B@state.ravel(),
                           h_l=np.r_[lo[self.variable], -cart_bounds],
                           h_u=np.r_[hi[self.variable], cart_bounds], x_l=xlo, x_u=xhi)
        solve_start = time.perf_counter()
        status = self.solver.solve()
        solve_ms = (time.perf_counter()-solve_start)*1000
        if status != piqp.PIQP_SOLVED:
            raise RuntimeError(f"QP failed: {status}")
        x = self.solver.result.x.reshape(n, 4, 7)
        if not np.isfinite(x).all():
            raise RuntimeError("nonfinite QP output")
        a = np.vstack([state[2], x[:, 2]])
        # Remove tiny terminal equality residuals before polynomial integration.
        # Larger residuals are failures, never silently repaired.
        terminal_v = state[1]+h*(a[0]/2+a[1:-1].sum(axis=0)+a[-1]/2)
        if max(np.max(abs(terminal_v)), np.max(abs(a[-1]))) > 1e-7:
            raise RuntimeError("terminal equality residual too large")
        a[-1] = 0
        terminal_v = state[1]+h*(a[0]/2+a[1:-1].sum(axis=0))
        a[-2] -= terminal_v/h
        states = np.empty((n+1, 3, 7))
        states[0] = state
        for k in range(n):
            q, v, ak = states[k]
            d = a[k+1]-ak
            states[k+1] = [q+h*v+h*h*ak/2+h*h*d/6, v+h*ak+h*d/2, a[k+1]]
        states[-1, 1:] = 0
        plan = Plan(states, h, solve_ms=solve_ms, iterations=self.solver.result.info.iter)
        states.flags.writeable = False
        return Plan(states, h, (time.perf_counter()-begin)*1000, solve_ms, plan.iterations)
