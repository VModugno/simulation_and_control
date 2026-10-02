"""HRP-4 whole-body walking controller (IS-MPC + QP inverse dynamics), pinocchio port.

Port of the DART IS-MPC demo to the RoboEnv pinocchio/pybullet stack.

Key differences from the original demo (and from the previous broken skeleton):

- pybullet base pose/velocity getters return the INERTIAL frame: retrieve_state()
  converts to the LINK frame (T_link = T_inertial * T_li^-1) and rotates the base
  velocity to the pin free-flyer LOCAL convention (see probe-verified formulas).
- The Kalman filter (demo simulation.py) IS ported: it runs before every
  mpc.solve on the com/zmp xy state, because the raw pybullet ZMP is noisy
  (sole-edge rocking, +-30% fz spikes) and feeding it unfiltered into the
  unstable LIP dynamics diverged laterally during in-place stepping (see the
  MS1 debug notes in NEXT_STEPS.md). The CoM position/velocity alone are
  exact from pinocchio; only the ZMP feedback needed the filter.
- The ZMP is aggregated from world-frame pybullet contact points (the joint
  reaction forces are unreliable while airborne). The tangential friction term
  of the demo formula is dropped: it scales with (zmp_z - point_z) ~ 0 on flat
  ground. When the total normal force is negligible the previous ZMP is held
  instead of resetting to zero (fixes the demo FIXME). On the very first
  retrieve_state (constructor, before any pybullet step) there are no contact
  points yet, so the ZMP is seeded with the CoM projection: at rest ZMP = CoM
  xy. A zero seed would make the MPC see a fake 3.5 cm offset and push the
  robot forward until the QP turned infeasible.
- The desired com height is pinned to the INITIAL measured com z (0.761 m)
  instead of the LIP constant h = 0.72 m: commanding 0.72 against an actual
  0.761 would pull the whole body down through the QP all the time.
- A guard holds the previous torque if the MPC solution explodes (|lip|>100):
  the osqp fallback can return huge-but-finite garbage instead of raising when
  it hits max_iter, which would otherwise flow straight into the references.
- The pinocchio joint order differs from the motor (ext) order:
  ext = [R_HIP_Y ... L_ELBOW_P], pin = [CHEST_P, CHEST_Y, L_SHOULDER_*, ...].
  inverse_dynamics pads current['joint'] raw into pin nv[6:], so the joint task
  vectors must be passed PIN-ORDERED (ReoderJoints2PinVec slices).
- The MPC tick (params['world_time_step'] = 0.01 s, 100 Hz) is intentionally
  slower than the simulation step (0.001 s, 1 kHz): the test loop re-applies the
  last torque command at every simulation step.
- t_max caps the tick at (start of the last planned step - 1) instead of the
  demo's horizon safety margin: freezing earlier left the final step half
  prepared and the robot tipped over the stance toe while the ZMP reference
  kept advancing inside the frozen horizon (probe-verified).

Control flow per 100 Hz tick (mirrors the demo customPreStep):
  retrieve_state -> mpc.solve -> desired com/zmp -> foot trajectories ->
  torso/base orientation reference -> ID-QP torques -> clip +-100 N*m.
"""

import copy

import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag

from .humanoid_controller.ismpc import Ismpc
from .humanoid_controller.footstep_planner import FootstepPlanner
from .humanoid_controller.foot_trajectory_generator import FootTrajectoryGenerator
from .humanoid_controller.inverse_dynamics import InverseDynamics
from .humanoid_controller.logger import Logger
from .humanoid_controller.filter import KalmanFilter


class Hrp4Controller:

    def __init__(self, dyn_model, sim, vref=None, use_gui=False):
        """
        Args:
            dyn_model: PinWrapper of the floating-base hrp4 model.
            sim: pybullet SimInterface (single robot).
            vref: gait reference, list of (vx, vy, wtheta) tuples, one per step.
                  Defaults to the walk gait of the IS-MPC demo.
            use_gui: enable the live desired/current CoM-ZMP logger plot.
        """
        self.sim = sim
        self.dyn_model = dyn_model
        self.pybullet_client = sim.GetPyBulletClient()
        self.robot_id = sim.bot[0].bot_pybullet
        self.use_gui = use_gui

        # --- parameters (IS-MPC demo values) ---
        # world_time_step is the 100 Hz CONTROL tick, NOT the 1 kHz sim step.
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'ss_duration': 70,
            'ds_duration': 30,
            'world_time_step': 0.01,
            'first_swing': 'right',
            'mu': 0.5,
            'N': 100,
            'dof': dyn_model.getNumberofActuatedJoints(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        self.mass = pin.computeTotalMass(dyn_model.pin_model)
        self.n_joints = self.params['dof']
        self.prev_zmp = np.zeros(3)

        # --- state ---
        self.time = 0  # control tick (100 Hz), not simulation steps
        self.initial = self.retrieve_state()
        self.current = copy.deepcopy(self.initial)
        self.desired = copy.deepcopy(self.initial)
        # support foot passed to the ID during the initial double support
        self.contact = 'lsole' if self.params['first_swing'] == 'right' else 'rsole'
        self.prev_zmp = np.array(self.initial['zmp']['pos'])

        # The LIP height h is a model constant, but the robot crouches lower
        # than the nominal h = 0.72 m. A constant 4 cm CoM-height error gives
        # the com task a steady downward pull that compounds the ZMP noise.
        self.com_height = np.array(self.initial['com']['pos'])[2]

        # --- whole-body inverse dynamics QP ---
        redundant_dofs = [
            "NECK_Y", "NECK_P",
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P",
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P",
        ]
        self.id = InverseDynamics(
            dyn_model, redundant_dofs,
            foot_size=self.params['foot_size'], mu=self.params['mu'])

        # --- gait reference ---
        if vref is None:
            vref = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10

        # footstep planner wants the initial 6d sole poses [rotvec, position]
        self.footstep_planner = FootstepPlanner(
            vref,
            self.initial['lsole']['pos'],
            self.initial['rsole']['pos'],
            self.params)

        self.mpc = Ismpc(self.initial, self.footstep_planner, self.params)

        self.foot_trajectory_generator = FootTrajectoryGenerator(
            self.initial, self.footstep_planner, self.params)

        # plan[last] is only the LANDING TARGET of the final swing: step `last`
        # itself would interpolate toward plan[last+1], which does not exist
        # (IndexError in the trajectory generator). The walk therefore ends when
        # the last step STARTS: get_start_time(last) is the first tick of that
        # phantom step, so freeze one tick earlier, in the double support that
        # follows the final swing. There the feet are planted at their final
        # poses and the MPC moving constraint saturates at the final footstep
        # (sigma clips to 1 past its window), i.e. the robot simply stands.
        # Freezing one step earlier (start(last) - N) leaves the ZMP reference
        # still advancing inside the frozen horizon while the feet are pinned
        # behind it: the robot tips forward over the stance toe and never lands
        # the final step (probe-verified failure mode).
        last = len(self.footstep_planner.footstep_plan) - 1
        self.t_max = self.footstep_planner.get_start_time(last) - 1

        # --- Kalman filter on the LIP state (demo wiring) ---
        # x = [com_x, vel_x, zmp_x, com_y, vel_y, zmp_y]; the predict uses the
        # previous MPC zmp rate, the update fuses the exact pinocchio CoM with
        # the noisy contact-based ZMP measurement.
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x0 = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0],
                       self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1]])
        self.kf = KalmanFilter(block_diag(A, A), block_diag(B, B), block_diag(H, H),
                               block_diag(Q, Q), block_diag(R, R), block_diag(P, P), x0)

        # --- torque command (ext/motor order) ---
        self.n_joints = dyn_model.getNumberofActuatedJoints()
        self.tau_cmd = np.zeros(self.n_joints)

        # --- logger (the raw q/dq entries are not loggable dictionaries) ---
        self.logger = Logger(self._loggable(self.initial))
        if self.use_gui:
            self.logger.initialize_plot()

    # ------------------------------------------------------------------ state

    @staticmethod
    def _loggable(state):
        """Filter out the raw q/dq arrays: Logger expects dict-of-dicts only."""
        return {k: v for k, v in state.items() if isinstance(v, dict)}

    def _base_link_state(self):
        """pybullet reports the base INERTIAL frame; pinocchio wants the LINK frame.

        T_link = T_inertial * T_li^-1 with T_li the (constant) local inertial
        transform. Velocity: world velocity of the inertial origin transferred to
        the link origin, then rotated into the link frame (pin free-flyer LOCAL).
        """
        sim = self.sim
        pos_i = np.array(sim.GetBasePosition(0))
        quat_i = np.array(sim.GetBaseOrientation(0))  # x y z w
        info = sim.getDynamicsInfo(self.robot_id, -1)
        p_li = np.array(info[3])
        q_li = np.array(info[4])

        R_i = pin.Quaternion(quat_i[3], quat_i[0], quat_i[1], quat_i[2]).matrix()
        R_li = pin.Quaternion(q_li[3], q_li[0], q_li[1], q_li[2]).matrix()
        R_link = R_i @ R_li.T
        p_link = pos_i - R_link @ p_li
        quat_link = pin.Quaternion(R_link).coeffs()  # x y z w

        # getBaseVelocity returns the inertial-frame origin velocity, world axes
        v_lin_i = np.array(sim.GetBaseLinVelocity(0))
        w_world = np.array(sim.GetBaseAngVelocity(0))
        v_lin_link_world = v_lin_i + np.cross(w_world, p_link - pos_i)
        v_local = R_link.T @ v_lin_link_world
        w_local = R_link.T @ w_world

        return p_link, quat_link, np.concatenate([v_local, w_local])

    def _compute_zmp(self, com_pos, l_sole_pos, r_sole_pos):
        """ZMP from world-frame pybullet contact points.

        The tangential friction term of the demo formula is dropped: it scales
        with (zmp_z - contact_z) ~ 0 on flat ground. If the normal force is
        negligible the previous ZMP is held (the demo reset it to zero, which
        would slam the MPC state back to the origin mid-swing).
        """
        fz = 0.
        zmp_xy = np.zeros(2)
        for c in self.pybullet_client.getContactPoints(bodyA=self.robot_id):
            if c[2] == self.robot_id:
                continue  # self contact
            f = c[9]  # normal force magnitude
            p = np.array(c[5])  # contact point on the robot, world frame
            fz += f
            zmp_xy += p[:2] * f

        if fz > 0.1:
            zmp = np.array([
                zmp_xy[0] / fz,
                zmp_xy[1] / fz,
                com_pos[2] - fz * self.params['h'] / (self.mass * self.params['g']),
            ])
        elif not np.any(self.prev_zmp):
            # Cold start: the constructor reads the state before the first
            # pybullet step, so no contacts exist yet and prev_zmp is still
            # the all-zero sentinel. At rest the ZMP is the CoM projected
            # toward the ground; seeding the hold with zeros made the MPC see
            # a fake 3.5 cm ZMP offset that pushed the robot forward until
            # the QP went infeasible.
            zmp = np.array([com_pos[0], com_pos[1],
                            com_pos[2] - self.params['h']])
        else:
            zmp = np.array(self.prev_zmp)

        # clip around the feet midpoint (the demo had a midpoint typo: (l+l)/2)
        midpoint = (l_sole_pos + r_sole_pos) / 2.
        zmp = np.clip(zmp, midpoint - 0.3, midpoint + 0.3)
        self.prev_zmp = np.array(zmp)
        return zmp

    def retrieve_state(self):
        sim = self.sim
        dyn = self.dyn_model

        # motor state (ext/motor order)
        q_mot = np.array(sim.GetMotorAngles(0))
        qd_mot = np.array(sim.GetMotorVelocities(0))

        p_link, quat_link, base_vel = self._base_link_state()

        # full state in ext convention (joint part ext-ordered; the wrapper
        # reorders internally wherever pinocchio order is needed)
        q_full = np.concatenate([p_link, quat_link, q_mot])        # nq = 31
        dq_full = np.concatenate([base_vel, qd_mot])                # nv = 30
        dq_pin = dyn.ReoderJoints2PinVec(dq_full, 'vel')

        # soles: 6d pose [rotvec, position], 6d velocity [angular, linear]
        feet = {}
        for foot in ('lsole', 'rsole'):
            link = dyn.getFeetLinkName(foot)
            tr, R = dyn.ComputeFK(q_full, link)
            dyn.ComputeJacobian(q_full, link, 'local_global')
            vel_lwa = dyn.res.J @ dq_pin          # rows: [linear; angular]
            feet[foot] = {
                'pos': np.hstack([pin.log3(R), tr]),
                'vel': vel_lwa[[3, 4, 5, 0, 1, 2]],  # dart order [angular; linear]
                'acc': np.zeros(6),
            }

        com_pos = np.array(dyn.ComputeCoMPosition(q_full))
        com_vel = np.array(dyn.ComputeCoMVelocity(q_full, dq_full))

        # torso/base: orientation-only tasks (3d rotvec + 3d world angular vel)
        # state key 'base' maps to the pin 'body' frame (the root link of hrp4)
        upper = {}
        for state_key, pin_frame in (('torso', 'torso'), ('base', 'body')):
            _, R = dyn.ComputeFK(q_full, pin_frame)
            dyn.ComputeJacobian(q_full, pin_frame, 'local_global')
            upper[state_key] = {
                'pos': pin.log3(R),
                'vel': dyn.res.J[3:6, :] @ dq_pin,
                'acc': np.zeros(3),
            }

        # joint task vectors must be PIN-ordered: inverse_dynamics pads them raw
        # into pin nv[6:] while the ext motor order differs from the pin order.
        q_pin = dyn.ReoderJoints2PinVec(q_full, 'pos')
        joint = {
            'pos': q_pin[7:],   # nq = 7 (base) + 24 (joints)
            'vel': dq_pin[6:],  # nv = 6 (base) + 24 (joints)
            'acc': np.zeros(self.n_joints),
        }

        # ZMP from measured contacts (uses current soles for the clip midpoint)
        zmp = self._compute_zmp(com_pos, feet['lsole']['pos'][3:], feet['rsole']['pos'][3:])

        return {
            'lsole': feet['lsole'],
            'rsole': feet['rsole'],
            'com': {'pos': com_pos, 'vel': com_vel, 'acc': np.zeros(3)},
            'torso': upper['torso'],
            'base': upper['base'],
            'joint': joint,
            'zmp': {'pos': zmp, 'vel': np.zeros(3), 'acc': np.zeros(3)},
            # raw ext-convention state consumed by inverse_dynamics
            'q': q_full,
            'dq': dq_full,
        }

    # --------------------------------------------------------------- control

    def ComputeController(self):
        """One 100 Hz control tick: returns the torque command (ext order)."""
        self.current = self.retrieve_state()
        # freeze the tick before the plan tail (planner/mpc index plan[k+1])
        t = min(self.time, self.t_max)

        # --- Kalman filter: fuse the exact pinocchio CoM with the noisy ---
        # --- contact-based ZMP before feeding the MPC                     ---
        u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1]])
        self.kf.predict(u)
        z = np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0],
                      self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1]])
        x_flt, _ = self.kf.update(z)
        self.current['com']['pos'][0] = x_flt[0]
        self.current['com']['vel'][0] = x_flt[1]
        self.current['zmp']['pos'][0] = x_flt[2]
        self.current['com']['pos'][1] = x_flt[3]
        self.current['com']['vel'][1] = x_flt[4]
        self.current['zmp']['pos'][1] = x_flt[5]

        # --- CoM/ZMP reference from the MPC ---
        lip_state, contact = self.mpc.solve(self.current, t)
        # osqp can return raw infeasibility values (~2^31) instead of raising
        # when it hits max_iter; those pass np.isfinite. Any physically
        # implausible reference (faster than 100 m/s from the origin) means
        # the QP diverged.
        if np.any(np.abs(lip_state['com']['pos']) > 100.) \
                or np.any(np.abs(lip_state['zmp']['pos']) > 100.):
            print(f"[Hrp4Controller] MPC QP diverged at tick {self.time}; "
                  "holding previous torque command")
            self.time += 1
            return self.tau_cmd
        if contact == 'ds':
            pass  # keep the previous support foot
        elif contact == 'ssleft':
            self.contact = 'lsole'
        elif contact == 'ssright':
            self.contact = 'rsole'

        self.desired['com']['pos'] = lip_state['com']['pos']
        self.desired['com']['vel'] = lip_state['com']['vel']
        self.desired['com']['acc'] = lip_state['com']['acc']
        self.desired['zmp']['pos'] = lip_state['zmp']['pos']
        self.desired['zmp']['vel'] = lip_state['zmp']['vel']
        # The LIP model is planar: its com z is the constant h. Command the
        # measured initial height instead so the com task does not fight the
        # actual crouch of the robot.
        self.desired['com']['pos'][2] = self.com_height

        # --- foot swing trajectories ---
        feet_traj = self.foot_trajectory_generator.generate_feet_trajectories_at_time(t)
        for side, foot in (('left', 'lsole'), ('right', 'rsole')):
            self.desired[foot]['pos'] = feet_traj[side]['pos']
            self.desired[foot]['vel'] = feet_traj[side]['vel']
            self.desired[foot]['acc'] = feet_traj[side]['acc']

        # --- torso/base orientation reference: average of the feet rotations ---
        # (orientation-only task; a flat-footed gait keeps the rotvec blocks ~0)
        for frame in ('torso', 'base'):
            self.desired[frame]['pos'] = (self.desired['lsole']['pos'][:3]
                                           + self.desired['rsole']['pos'][:3]) / 2.
            self.desired[frame]['vel'] = (self.desired['lsole']['vel'][:3]
                                           + self.desired['rsole']['vel'][:3]) / 2.
            self.desired[frame]['acc'] = (self.desired['lsole']['acc'][:3]
                                           + self.desired['rsole']['acc'][:3]) / 2.

        # --- whole-body inverse dynamics QP (joint task holds the initial posture) ---
        tau = self.id.get_joint_torques(self.desired, self.current, contact)
        self.tau_cmd = np.clip(tau, -100., 100.)  # URDF effort limit

        # --- logging ---
        self.logger.log_data(self._loggable(self.desired), self._loggable(self.current))
        if self.use_gui and self.time % 10 == 0:
            self.logger.update_plot()

        self.time += 1
        return self.tau_cmd