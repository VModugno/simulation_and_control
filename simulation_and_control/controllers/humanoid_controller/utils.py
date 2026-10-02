import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_vector_difference(rotvec_a, rotvec_b):
    R_a = R.from_rotvec(rotvec_a)
    R_b = R.from_rotvec(rotvec_b)
    R_diff = R_b.inv() * R_a
    return R_diff.as_rotvec()


def pose_difference(pose_a, pose_b):
    pos_diff = pose_a[:3] - pose_b[:3]
    rot_diff = rotation_vector_difference(pose_a[3:], pose_b[3:])
    return np.hstack((pos_diff, rot_diff))


# converts a rotation matrix to a rotation vector
def get_rotvec(rot_matrix):
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_rotvec()


def block_diag(*arrays):
    arrays = [np.atleast_2d(a) if np.isscalar(a) else np.atleast_2d(a) for a in arrays]

    rows = sum(arr.shape[0] for arr in arrays)
    cols = sum(arr.shape[1] for arr in arrays)
    block_matrix = np.zeros((rows, cols), dtype=arrays[0].dtype)

    current_row = 0
    current_col = 0

    for arr in arrays:
        r, c = arr.shape
        block_matrix[current_row:current_row + r, current_col:current_col + c] = arr
        current_row += r
        current_col += c

    return block_matrix


_conic_available = None


def casadi_conic_available():
    # casadi wheels ship without conic solver plugins on some platforms
    # (windows confirmed); probe once and cache instead of guessing by os
    global _conic_available
    if _conic_available is None:
        try:
            import casadi
            casadi.conic('qp_backend_probe', 'osqp')
            _conic_available = True
        except Exception:
            print('casadi conic plugins unavailable, qp solving falls back to the osqp package')
            _conic_available = False
    return _conic_available


def _solve_qp_osqp(n_vars, H, F, A_eq, b_eq, A_ineq, b_ineq):
    import osqp
    from scipy import sparse

    A_rows = []
    l = []
    u = []
    if A_eq is not None and b_eq is not None:
        A_rows.append(np.asarray(A_eq, dtype=float))
        l.append(np.asarray(b_eq, dtype=float))
        u.append(np.asarray(b_eq, dtype=float))
    if A_ineq is not None and b_ineq is not None:
        A_rows.append(np.asarray(A_ineq, dtype=float))
        l.append(np.full(len(b_ineq), -np.inf))
        u.append(np.asarray(b_ineq, dtype=float))
    A = sparse.csc_matrix(np.vstack(A_rows)) if A_rows else sparse.csc_matrix((0, n_vars))
    H_arr = np.asarray(H, dtype=float)
    P = sparse.csc_matrix(0.5 * (H_arr + H_arr.T))

    solver = osqp.OSQP()
    solver.setup(P=P, q=np.asarray(F, dtype=float), A=A,
                 l=np.concatenate(l) if l else np.array([]),
                 u=np.concatenate(u) if u else np.array([]),
                 verbose=False)
    result = solver.solve()
    if result.x is None:
        raise RuntimeError(f"osqp status: {result.info.status}")
    return result.x


# solves a constrained QP; casadi where its conic plugins exist, osqp otherwise
class QPSolver:
    def __init__(self, n_vars, n_eq_constraints=0, n_ineq_constraints=0):
        self.n_vars = n_vars
        self.n_eq_constraints = n_eq_constraints
        self.n_ineq_constraints = n_ineq_constraints
        self.opti = None
        if casadi_conic_available():
            self._build_casadi()

    def _build_casadi(self):
        import casadi as ca

        self.opti = ca.Opti('conic')
        self.x = self.opti.variable(self.n_vars)

        self.F_ = self.opti.parameter(self.n_vars)
        self.H_ = self.opti.parameter(self.n_vars, self.n_vars)
        objective = 0.5 * self.x.T @ self.H_ @ self.x + self.F_.T @ self.x
        self.opti.minimize(objective)

        self.A_eq_ = self.opti.parameter(self.n_eq_constraints, self.n_vars)
        self.b_eq_ = self.opti.parameter(self.n_eq_constraints)
        if self.n_eq_constraints > 0:
            self.opti.subject_to(self.A_eq_ @ self.x == self.b_eq_)

        if self.n_ineq_constraints > 0:
            self.A_ineq_ = self.opti.parameter(self.n_ineq_constraints, self.n_vars)
            self.b_ineq_ = self.opti.parameter(self.n_ineq_constraints)
            self.opti.subject_to(self.A_ineq_ @ self.x <= self.b_ineq_)
        else:
            self.A_ineq_ = None
            self.b_ineq_ = None

        p_opts = {'expand': True}
        s_opts = {'max_iter': 1000, 'verbose': False}
        self.opti.solver('osqp', p_opts, s_opts)

    def set_values(self, H, F, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        if self.opti is None:
            self._H = np.asarray(H, dtype=float)
            self._F = np.asarray(F, dtype=float)
            self._A_eq = np.asarray(A_eq, dtype=float) if A_eq is not None else None
            self._b_eq = np.asarray(b_eq, dtype=float) if b_eq is not None else None
            self._A_ineq = np.asarray(A_ineq, dtype=float) if A_ineq is not None else None
            self._b_ineq = np.asarray(b_ineq, dtype=float) if b_ineq is not None else None
            return
        self.opti.set_value(self.H_, H)
        self.opti.set_value(self.F_, F)
        if self.n_eq_constraints > 0 and A_eq is not None and b_eq is not None:
            self.opti.set_value(self.A_eq_, A_eq)
            self.opti.set_value(self.b_eq_, b_eq)
        if self.n_ineq_constraints > 0 and A_ineq is not None and b_ineq is not None:
            self.opti.set_value(self.A_ineq_, A_ineq)
            self.opti.set_value(self.b_ineq_, b_ineq)

    def solve(self):
        try:
            if self.opti is None:
                return _solve_qp_osqp(self.n_vars, self._H, self._F,
                                     self._A_eq, self._b_eq, self._A_ineq, self._b_ineq)
            solution = self.opti.solve()
            return solution.value(self.x)
        except RuntimeError as e:
            print("QP Solver failed:", e)
            return np.zeros(self.n_vars)