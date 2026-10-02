import numpy as np

from .utils import QPSolver


class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.N = params['N']
    N = self.N
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.step_height = params['step_height']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.footstep_plan = self.footstep_planner.footstep_plan
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    # lip model matrices
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # flattened qp variables: X (6 x (N+1)) row-major, then U (2 x N) row-major
    self.n_vars = 6 * (N + 1) + 2 * N
    self.n_eq = 6 * N + 6 + 2  # dynamics + initial state + periodic tail stability
    self.n_ineq = 4 * N        # zmp box constraints on both axes
    self.qp = QPSolver(self.n_vars, self.n_eq, self.n_ineq)

    self.H = self._build_cost_matrix()
    self.A_eq = self._build_dynamics_matrix()
    self.A_ineq = self._build_zmp_box_matrix()
    self.b_eq = np.zeros(self.n_eq)
    self.b_ineq = np.zeros(self.n_ineq)

    self.x = np.zeros(6)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def _idx_x(self, j, i):
    return j * (self.N + 1) + i

  def _idx_u(self, a, i):
    return 6 * (self.N + 1) + a * self.N + i

  def _build_cost_matrix(self):
    N = self.N
    H = np.zeros((self.n_vars, self.n_vars))
    for a in range(2):
      for i in range(N):
        H[self._idx_u(a, i), self._idx_u(a, i)] = 2.  # sumsqr(u)
    for i in range(1, N + 1):
      H[self._idx_x(2, i), self._idx_x(2, i)] = 200.  # 100 * sumsqr(zmp_x - mc_x)
      H[self._idx_x(5, i), self._idx_x(5, i)] = 200.  # 100 * sumsqr(zmp_y - mc_y)
    return H

  def _build_dynamics_matrix(self):
    N = self.N
    delta = self.delta
    eta2 = self.eta ** 2
    A = np.zeros((self.n_eq, self.n_vars))

    row = 0
    for i in range(N):
      # x axis
      A[row, self._idx_x(0, i + 1)] = 1.
      A[row, self._idx_x(0, i)] -= 1.
      A[row, self._idx_x(1, i)] -= delta
      row += 1
      A[row, self._idx_x(1, i + 1)] = 1.
      A[row, self._idx_x(1, i)] -= 1.
      A[row, self._idx_x(0, i)] -= delta * eta2
      A[row, self._idx_x(2, i)] += delta * eta2
      row += 1
      A[row, self._idx_x(2, i + 1)] = 1.
      A[row, self._idx_x(2, i)] -= 1.
      A[row, self._idx_u(0, i)] -= delta
      row += 1
      # y axis
      A[row, self._idx_x(3, i + 1)] = 1.
      A[row, self._idx_x(3, i)] -= 1.
      A[row, self._idx_x(4, i)] -= delta
      row += 1
      A[row, self._idx_x(4, i + 1)] = 1.
      A[row, self._idx_x(4, i)] -= 1.
      A[row, self._idx_x(3, i)] -= delta * eta2
      A[row, self._idx_x(5, i)] += delta * eta2
      row += 1
      A[row, self._idx_x(5, i + 1)] = 1.
      A[row, self._idx_x(5, i)] -= 1.
      A[row, self._idx_u(1, i)] -= delta
      row += 1

    for j in range(6):
      A[row, self._idx_x(j, 0)] = 1.
      row += 1

    eta3 = self.eta ** 3
    A[row, self._idx_x(1, 0)] = 1.
    A[row, self._idx_x(0, 0)] = eta3
    A[row, self._idx_x(2, 0)] -= eta3
    A[row, self._idx_x(1, N)] -= 1.
    A[row, self._idx_x(0, N)] -= eta3
    A[row, self._idx_x(2, N)] += eta3
    row += 1
    A[row, self._idx_x(4, 0)] = 1.
    A[row, self._idx_x(3, 0)] = eta3
    A[row, self._idx_x(5, 0)] -= eta3
    A[row, self._idx_x(4, N)] -= 1.
    A[row, self._idx_x(3, N)] -= eta3
    A[row, self._idx_x(5, N)] += eta3
    row += 1
    return A

  def _build_zmp_box_matrix(self):
    N = self.N
    A = np.zeros((self.n_ineq, self.n_vars))
    row = 0
    for i in range(1, N + 1):
      A[row, self._idx_x(2, i)] = 1.
      row += 1
      A[row, self._idx_x(2, i)] = -1.
      row += 1
      A[row, self._idx_x(5, i)] = 1.
      row += 1
      A[row, self._idx_x(5, i)] = -1.
      row += 1
    return A

  def solve(self, current, t):
    self.x = np.array([current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
                       current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1]])

    mc_x, mc_y = self.generate_moving_constraint(t)
    N = self.N
    half = self.foot_size / 2.

    F = np.zeros(self.n_vars)
    for i in range(1, N + 1):
      F[self._idx_x(2, i)] = -200. * mc_x[i - 1]
      F[self._idx_x(5, i)] = -200. * mc_y[i - 1]

    self.b_eq[6 * N:6 * N + 6] = self.x

    row = 0
    for i in range(1, N + 1):
      self.b_ineq[row] = mc_x[i - 1] + half
      row += 1
      self.b_ineq[row] = half - mc_x[i - 1]
      row += 1
      self.b_ineq[row] = mc_y[i - 1] + half
      row += 1
      self.b_ineq[row] = half - mc_y[i - 1]
      row += 1

    z = self.qp_solve(F)
    self.x = np.array([z[self._idx_x(j, 1)] for j in range(6)])
    self.u = np.array([z[self._idx_u(0, 0)], z[self._idx_u(1, 0)]])

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.h])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], 0.])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], 0.])
    self.lip_state['zmp']['vel'] = np.hstack((self.u, 0.))
    self.lip_state['com']['acc'] = np.hstack((self.eta**2 * (self.lip_state['com']['pos'][:2] - self.lip_state['zmp']['pos'][:2]), 0.))

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact += self.footstep_planner.footstep_plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact

  def qp_solve(self, F):
    self.qp.set_values(self.H, F, self.A_eq, self.b_eq, self.A_ineq, self.b_ineq)
    return self.qp.solve()

  def generate_moving_constraint_at_time(self, time):
    step_index = self.footstep_planner.get_step_index_at_time(time)
    time_in_step = time - self.footstep_planner.get_start_time(step_index)
    phase = self.footstep_planner.get_phase_at_time(time)
    single_support_duration = self.footstep_plan[step_index]['ss_duration']
    double_support_duration = self.footstep_plan[step_index]['ds_duration']

    if phase == 'ss':
      return self.footstep_plan[step_index]['pos']

    # linear interpolation for x and y coordinates of the foot positions during double support
    if step_index == 0: start_pos = (self.initial['lsole']['pos'][3:] + self.initial['rsole']['pos'][3:]) / 2.
    else:               start_pos = np.array(self.footstep_plan[step_index]['pos'])
    target_pos = np.array(self.footstep_plan[step_index + 1]['pos'])

    moving_constraint = start_pos + (target_pos - start_pos) * ((time_in_step - single_support_duration) / double_support_duration)
    return moving_constraint

  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lsole']['pos'][3] + self.initial['rsole']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lsole']['pos'][4] + self.initial['rsole']['pos'][4]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_plan[j]['ds_duration']
      fs_current_pos = self.footstep_plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y