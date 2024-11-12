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

class Hrp4Controller:
    def __init__(self,dyn_model):
        
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

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()


        # set initial configuration
        #initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
        #                         'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
        #                         'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
        #                         'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
        #                         'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}
        

        # position the robot on the ground
        #lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        #rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        #self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        #self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        #self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

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
            self.initial['lsole']['pos'],
            self.initial['rsole']['pos'],
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
        
    

