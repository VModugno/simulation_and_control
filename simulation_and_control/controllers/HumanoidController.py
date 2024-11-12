import numpy as np
import pin_wrapper as dyn_model
import copy
from utils import *
import os
import humanoid_controller.ismpc as ismpc
import humanoid_controller.footstep_planner as footstep_planner
import humanoid_controller.inverse_dynamics as id
import humanoid_controller.filter as filter
import humanoid_controller.foot_trajectory_generator as ftg

from humanoid_controller.logger import Logger
from scipy.spatial.transform import Rotation as R  # For quaternion to rotation vector conversion
import pinocchio as pin

class Hrp4Controller:
    def __init__(self,dyn_model,sim):
        
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'ss_duration': 70,
            'ds_duration': 30,
            'world_time_step': dyn_model.getTimeStep(),
            'first_swing': 'right',
            'µ': 0.5,
            'N': 100,
            'dof':dyn_model.getNumberofActuatedJoints(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

        self.dyn_model = dyn_model
        init_motor_angles = sim.GetMotorAngles(0)
        init_robot_pos = sim.GetBasePosition()
        init_robot_ori = sim.GetBaseOrientation()
        init_q = np.array([init_robot_pos, init_robot_ori, init_motor_angles])
        left_foot_link_name = dyn_model.GetLinkName('FL')
        left_foot_pos,_ = dyn_model.ComputeFK(init_q,left_foot_link_name)
        right_foot_link_name = dyn_model.GetLinkName('FR')
        right_foot_pos,_ = dyn_model.ComputeFK(init_q,right_foot_link_name)
        # robot links
        #self.lsole = hrp4.getBodyNode('l_sole')
        #self.rsole = hrp4.getBodyNode('r_sole')
        #self.torso = hrp4.getBodyNode('torso')
        #self.base  = hrp4.getBodyNode('body')

       

        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lsole' if self.params['first_swing'] == 'right' else 'rsole' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

        # initialize footstep planner
        reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10
        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            left_foot_pos,
            right_foot_pos,
            self.params
            )

        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1]])
        self.kf = filter.KalmanFilter(block_diag(A, A), \
                                      block_diag(B, B), \
                                      block_diag(H, H), \
                                      block_diag(Q, Q), \
                                      block_diag(R, R), \
                                      block_diag(P, P), \
                                      x)

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot()


    def retrieve_state(self,sim):
        
        index = 0  # Assuming a single robot instance
        dyn_model = self.dyn_model  # Access the dynamic model
        

        # Get motor angles and velocities
        motor_angles = sim.GetMotorAngles(index)
        motor_velocities = sim.GetMotorVelocities(index)
        # Assuming motor accelerations are available (else need to add GetMotorAccelerations to sim)
        motor_accelerations = sim.GetMotorAccelerationTMinusOne(index)

        # Get base position and orientation
        base_position = sim.GetBasePosition(index)  # [x, y, z]
        base_orientation_quat = sim.GetBaseOrientation(index)  # [x, y, z, w]

        # Convert base orientation quaternion to rotation vector
        base_orientation = R.from_quat(base_orientation_quat).as_rotvec()  # [rx, ry, rz]

        # Get base linear and angular velocities
        base_lin_velocity = sim.GetBaseLinVelocity(index)  # [vx, vy, vz]
        base_ang_velocity = sim.GetBaseAngVelocity(index)  # [wx, wy, wz]

        # Build the full state vectors (configuration and velocity)
        q_base = np.concatenate([base_position, base_orientation_quat])  # Base position and orientation (quaternion)
        dq_base = np.concatenate([base_lin_velocity, base_ang_velocity])  # Base linear and angular velocities

        # Assemble the full state vectors including joints
        q_full = np.concatenate([q_base, motor_angles])
        dq_full = np.concatenate([dq_base, motor_velocities])

        # Initialize dictionaries to store foot states
        foot_states = {}
        foot_names = sim.bot[index].foot_link_ids.keys()

        # For each foot, compute position and velocity using dyn_model
        for foot_name in foot_names:
            # Get the link name for the foot
            foot_link_name = dyn_model.GetLinkName(foot_name)

            # Compute forward kinematics to get foot position and orientation
            foot_translation, foot_rotation_matrix = dyn_model.ComputeFK(q_full, foot_link_name)
            foot_position = foot_translation  # Position in world frame
            foot_orientation_matrix = foot_rotation_matrix  # Rotation matrix in world frame
            foot_orientation = pin.log3(foot_orientation_matrix)  # Convert to rotation vector
            foot_pose = np.hstack((foot_orientation, foot_position))

            # Compute Jacobian to get foot velocities
            res = dyn_model.ComputeJacobian(q_full, foot_link_name, local_or_global='global')
            foot_jacobian = res.J  # Extract the Jacobian matrix

            # Reorder dq_full to Pinocchio's expected order
            dq_full_pin = dyn_model.ReoderJoints2PinVec(dq_full, "vel")

            # Compute foot velocity
            foot_velocity = foot_jacobian @ dq_full_pin  # Spatial velocity

            # Store in the dictionary
            foot_states[foot_name] = {
                'pos': foot_pose,
                'vel': foot_velocity,
                'acc': np.zeros(6)  # Placeholder, as accelerations are not directly available
            }

        # Get torso position, orientation, and velocities using dyn_model
        torso_link_name = 'torso'  # Adjust based on your robot's link naming
        # Compute FK for the torso
        torso_translation, torso_rotation_matrix = dyn_model.ComputeFK(q_full, torso_link_name)
        torso_position = torso_translation
        torso_orientation_matrix = torso_rotation_matrix
        torso_orientation = pin.log3(torso_orientation_matrix)

        # Compute torso velocity
        res_torso = dyn_model.ComputeJacobian(q_full, torso_link_name, local_or_global='global')
        torso_jacobian = res_torso.J

        # Reuse dq_full_pin
        torso_velocity = torso_jacobian @ dq_full_pin
        torso_linear_velocity = torso_velocity[:3]
        torso_angular_velocity = torso_velocity[3:]

        # Compute Center of Mass (CoM) position and velocity using dyn_model
        # Assuming ComputeCoMPosition and ComputeCoMVelocity internally handle reordering
        com_position = dyn_model.ComputeCoMPosition(q_full)
        com_velocity = dyn_model.ComputeCoMVelocity(q_full, dq_full)

        # Compute ground reaction forces (GRF) at the feet
        foot_grfs = sim.ComputeFootGRF(index)  # Returns a dictionary {foot_name: force_vector}

        # Compute ZMP
        total_force = np.zeros(3)
        zmp_numerator = np.zeros(3)
        for foot_name, force in foot_grfs.items():
            foot_pos = foot_states[foot_name]['pos'][3:]  # Extract position part
            force = np.array(force)
            if np.linalg.norm(force) < 0.1:
                continue  # Ignore negligible forces
            total_force += force
            zmp_numerator += foot_pos * force[2]

        if total_force[2] > 0.1:
            zmp = zmp_numerator / total_force[2]
        else:
            zmp = np.zeros(3)  # Robot is in the air or not enough contact force

        # Clip ZMP close to robot's feet positions
        # Compute midpoint between feet
        foot_positions = np.array([foot_states[foot_name]['pos'][3:] for foot_name in foot_states])
        midpoint = np.mean(foot_positions, axis=0)
        zmp = np.clip(zmp, midpoint - 0.3, midpoint + 0.3)

        # Assemble the state dictionary
        state = {
            'lfoot': foot_states.get('FL', {'pos': np.zeros(6), 'vel': np.zeros(6), 'acc': np.zeros(6)}),
            'rfoot': foot_states.get('FR', {'pos': np.zeros(6), 'vel': np.zeros(6), 'acc': np.zeros(6)}),
            'com': {
                'pos': com_position,
                'vel': com_velocity,
                'acc': np.zeros(3)  # Placeholder
            },
            'torso': {
                'pos': torso_orientation,
                'vel': torso_angular_velocity,
                'acc': np.zeros(3)  # Placeholder
            },
            'base': {
                'pos': base_orientation,  # Base orientation as rotation vector
                'vel': base_ang_velocity,
                'acc': np.zeros(3)  # Placeholder
            },
            'joint': {
                'pos': q_full,
                'vel': dq_full,
                'acc': np.concatenate((np.zeros(6), motor_accelerations))  # Base accelerations set to zero
            },
            'zmp': {
                'pos': zmp,
                'vel': np.zeros(3),  # Placeholder
                'acc': np.zeros(3)   # Placeholder
            }
        }

        return state


    def UpdateEstimatedState(self,u):
        # update kalman filter
         #u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1]])
         self.kf.predict(u)
         x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
                                             self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1]]))
        
         # update current state
         self.current['com']['pos'][0] = x_flt[0]
         self.current['com']['vel'][0] = x_flt[1]
         self.current['zmp']['pos'][0] = x_flt[2]
         self.current['com']['pos'][1] = x_flt[3]
         self.current['com']['vel'][1] = x_flt[4]
         self.current['zmp']['pos'][1] = x_flt[5]

    
    def ComputeController(self):
        #     # get references using MPC
    #     #self.desired['com']['pos'] = np.array([0., 0., 0.75])
    #     lip_state, contact = self.mpc.solve(self.current, self.time)
    #     if contact == 'ds':
    #         pass
    #     elif contact == 'ssleft':
    #         self.contact = 'lsole'
    #     elif contact == 'ssright':
    #         self.contact = 'rsole'

    #     self.desired['com']['pos'] = lip_state['com']['pos']
    #     self.desired['com']['vel'] = lip_state['com']['vel']
    #     self.desired['com']['acc'] = lip_state['com']['acc']
    #     self.desired['zmp']['pos'] = lip_state['zmp']['pos']
    #     self.desired['zmp']['vel'] = lip_state['zmp']['vel']

    #     # get foot trajectories
    #     feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
    #     self.desired['lsole']['pos'] = feet_trajectories['left']['pos']
    #     self.desired['lsole']['vel'] = feet_trajectories['left']['vel']
    #     self.desired['lsole']['acc'] = feet_trajectories['left']['acc']
    #     self.desired['rsole']['pos'] = feet_trajectories['right']['pos']
    #     self.desired['rsole']['vel'] = feet_trajectories['right']['vel']
    #     self.desired['rsole']['acc'] = feet_trajectories['right']['acc']

    #     # set torso and base references to the average of the feet
    #     self.desired['torso']['pos'] = (self.desired['lsole']['pos'][:3] + self.desired['rsole']['pos'][:3]) / 2.
    #     self.desired['torso']['vel'] = (self.desired['lsole']['vel'][:3] + self.desired['rsole']['vel'][:3]) / 2.
    #     self.desired['torso']['acc'] = (self.desired['lsole']['acc'][:3] + self.desired['rsole']['acc'][:3]) / 2.
    #     self.desired['base']['pos']  = (self.desired['lsole']['pos'][:3] + self.desired['rsole']['pos'][:3]) / 2.
    #     self.desired['base']['vel']  = (self.desired['lsole']['vel'][:3] + self.desired['rsole']['vel'][:3]) / 2.
    #     self.desired['base']['acc']  = (self.desired['lsole']['acc'][:3] + self.desired['rsole']['acc'][:3]) / 2.

    #     # get torque commands using inverse dynamics
    #     commands = self.id.get_joint_torques(self.desired, self.current, contact)
        
    

