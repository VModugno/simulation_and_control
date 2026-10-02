import numpy as np
import pinocchio as pin

from .utils import QPSolver, rotation_vector_difference


class InverseDynamics:
    """Whole-body inverse dynamics QP (ported from the DART IS-MPC demo to pinocchio).

    Soles for [q_ddot(nv), tau(nv), f_c(12)] subject to the floating-base dynamics
    equality, friction-cone/CoP inequalities on the sole contact forces, and the
    weighted task costs (feet, com, torso/base orientation, redundant joints).
    """

    def __init__(self, dyn_model, redundant_dofs, foot_size=0.1, mu=0.5):
        self.dyn_model = dyn_model
        self.model = dyn_model.pin_model
        self.data = dyn_model.pin_data
        self.dofs = self.model.nv
        self.d = foot_size / 2.
        self.mu = mu

        # sole frame ids from the pin wrapper feet table ('lsole'/'rsole' -> l_sole/r_sole frames)
        self.lsole_id = dyn_model.feet_id['lsole']
        self.rsole_id = dyn_model.feet_id['rsole']
        self.torso_id = self.model.getFrameId('torso')
        # base task frame = pin root link ('body' for hrp4); the config floating_base_name is not a pin frame
        self.base_id = self.model.getFrameId('body')

        # sizes for the QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims
        self.n_eq_constraints = self.dofs
        self.n_ineq_constraints = 8 * self.num_contacts

        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs (velocity space, pin ordering)
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            if joint.nq == 0:
                continue
            name = self.model.names[i]
            if name in redundant_dofs:
                for k in range(joint.nv):
                    self.joint_selection[joint.idx_v + k, joint.idx_v + k] = 1.

    def get_joint_torques(self, desired, current, contact):
        contact_l = contact == 'ssleft'  or contact == 'ds'
        contact_r = contact == 'ssright' or contact == 'ds'

        # pin state (ext order in, reordering handled by the wrapper)
        q = self.dyn_model.ReoderJoints2PinVec(current['q'], 'pos')
        v = self.dyn_model.ReoderJoints2PinVec(current['dq'], 'vel')
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.dofs))
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q, v, np.zeros(self.dofs))
        pin.computeJointJacobians(self.model, self.data, q)

        # dynamics terms (pin ordering): M q_ddot + c + g = S tau + Jc^T f_c
        self.dyn_model.ComputeAllTerms(current['q'], current['dq'])
        inertia_matrix = np.array(self.dyn_model.res.M)
        n_term = np.array(self.dyn_model.res.c) + np.array(self.dyn_model.res.g)

        # frame jacobians (local world aligned, pin rows [lin; ang] -> dart rows [ang; lin])
        def frame_J_dart(frame_id):
            J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
            return J[[3, 4, 5, 0, 1, 2], :]

        J_lsole = frame_J_dart(self.lsole_id)
        J_rsole = frame_J_dart(self.rsole_id)
        J_torso_ang = pin.getFrameJacobian(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)[3:6, :]
        J_base_ang = pin.getFrameJacobian(self.model, self.data, self.base_id, pin.LOCAL_WORLD_ALIGNED)[3:6, :]
        J_com = pin.jacobianCenterOfMass(self.model, self.data, q)

        # jacobian drift terms Jdot @ v (classical accelerations at zero q_ddot)
        acc_lsole = pin.getFrameClassicalAcceleration(self.model, self.data, self.lsole_id, pin.LOCAL_WORLD_ALIGNED)
        acc_rsole = pin.getFrameClassicalAcceleration(self.model, self.data, self.rsole_id, pin.LOCAL_WORLD_ALIGNED)
        acc_torso = pin.getFrameClassicalAcceleration(self.model, self.data, self.torso_id, pin.LOCAL_WORLD_ALIGNED)
        acc_base = pin.getFrameClassicalAcceleration(self.model, self.data, self.base_id, pin.LOCAL_WORLD_ALIGNED)
        Jdotv = {'lsole': np.concatenate([acc_lsole.angular, acc_lsole.linear]),
                 'rsole': np.concatenate([acc_rsole.angular, acc_rsole.linear]),
                 'com':   np.array(self.data.acom[0]),
                 'torso': np.array(acc_torso.angular),
                 'base':  np.array(acc_base.angular),
                 'joints': np.zeros(self.dofs)}

        # full pin-v-space copies of the joint task vectors (pad actuated block into nv, base rows zero)
        joint_pos_nv = np.zeros(self.dofs)
        joint_vel_nv = np.zeros(self.dofs)
        joint_acc_nv = np.zeros(self.dofs)
        joint_pos_nv_des = np.zeros(self.dofs)
        joint_vel_nv_des = np.zeros(self.dofs)
        joint_pos_nv[6:] = current['joint']['pos']
        joint_vel_nv[6:] = current['joint']['vel']
        joint_pos_nv_des[6:] = desired['joint']['pos']
        joint_vel_nv_des[6:] = desired['joint']['vel']
        joint_acc_nv[6:] = desired['joint']['acc']

        # weights and gains (from the original demo)
        tasks = ['lsole', 'rsole', 'com', 'torso', 'base', 'joints']
        weights   = {'lsole':  1., 'rsole':  1., 'com':  1., 'torso': 1., 'base': 1., 'joints': 1.e-2}
        pos_gains = {'lsole': 10., 'rsole': 10., 'com':  5., 'torso': 1., 'base': 1., 'joints': 10.}
        vel_gains = {'lsole': 10., 'rsole': 10., 'com': 10., 'torso': 2., 'base': 2., 'joints': 1.e-1}

        J = {'lsole': J_lsole, 'rsole': J_rsole, 'com': J_com, 'torso': J_torso_ang,
             'base': J_base_ang, 'joints': self.joint_selection}

        # feedforward terms
        ff = {'lsole': desired['lsole']['acc'],
              'rsole': desired['rsole']['acc'],
              'com':   desired['com']['acc'],
              'torso': desired['torso']['acc'],
              'base':  desired['base']['acc'],
              'joints': joint_acc_nv}

        # error vectors (dart 6d stacking: pose [rotvec, pos], spatial vel [ang, lin])
        pos_error = {'lsole': np.concatenate([rotation_vector_difference(desired['lsole']['pos'][:3], current['lsole']['pos'][:3]),
                                              desired['lsole']['pos'][3:] - current['lsole']['pos'][3:]]),
                     'rsole': np.concatenate([rotation_vector_difference(desired['rsole']['pos'][:3], current['rsole']['pos'][:3]),
                                              desired['rsole']['pos'][3:] - current['rsole']['pos'][3:]]),
                     'com':   desired['com']['pos'] - current['com']['pos'],
                     'torso': rotation_vector_difference(desired['torso']['pos'], current['torso']['pos']),
                     'base':  rotation_vector_difference(desired['base']['pos'], current['base']['pos']),
                     'joints': joint_pos_nv_des - joint_pos_nv}

        vel_error = {'lsole': desired['lsole']['vel'] - current['lsole']['vel'],
                     'rsole': desired['rsole']['vel'] - current['rsole']['vel'],
                     'com':   desired['com']['vel']   - current['com']['vel'],
                     'torso': desired['torso']['vel'] - current['torso']['vel'],
                     'base':  desired['base']['vel']  - current['base']['vel'],
                     'joints': joint_vel_nv_des}

        # cost function
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            H_task = weights[task] * J[task].T @ J[task]
            F_task = - weights[task] * J[task].T @ (ff[task]
                                                    + vel_gains[task] * vel_error[task]
                                                    + pos_gains[task] * pos_error[task]
                                                    - Jdotv[task])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # regularization of contact forces
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * 1e-6

        # dynamics equality: M q_ddot - S tau - Jc^T f_c = -(c + g)
        actuation_matrix = np.zeros((self.dofs, self.dofs))
        actuation_matrix[6:, 6:] = np.eye(self.dofs - 6)
        contact_jacobian = np.vstack((contact_l * J_lsole, contact_r * J_rsole))
        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - contact_jacobian.T))
        b_eq = - n_term

        # friction cone + CoP inequalities (dart spatial force stacking [torque(3), force(3)])
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        A = np.array([[ 1, 0, 0, 0, 0, -self.d],
                      [-1, 0, 0, 0, 0, -self.d],
                      [0,  1, 0, 0, 0, -self.d],
                      [0, -1, 0, 0, 0, -self.d],
                      [0, 0, 0,  1, 0, -self.mu],
                      [0, 0, 0, -1, 0, -self.mu],
                      [0, 0, 0, 0,  1, -self.mu],
                      [0, 0, 0, 0, -1, -self.mu]])
        from .utils import block_diag
        A_ineq[0:self.n_ineq_constraints, f_c_indices] = block_diag(A, A)

        # solve
        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        tau = solution[tau_indices]
        # reorder pin joint torques to the pybullet actuator order
        tau_ext = self.dyn_model.ReoderJoints2ExtVec(tau, 'vel')
        return np.asarray(tau_ext)[6:]