import numpy as np
import time
# TODO replace this with_pin_wrapper
from gym_quadruped.utils.quadruped_utils import LegsAttr
#from quadruped_pympc import config as cfg

from foothold_reference_generator import FootholdReferenceGenerator
from periodic_gait_generator import PeriodicGaitGenerator
from swing_trajectory_controller import SwingTrajectoryController
from terrain_estimator import TerrainEstimator




class WBInterface:
    """
    WBInterface is responsible for interfacing with the whole body controller of a quadruped robot.
    It initializes the necessary components for motion planning and control, including gait generation,
    swing trajectory control, and terrain estimation.
    """

    def __init__(self,
                 initial_feet_pos: LegsAttr,
                 legs_order: tuple[str, str, str, str] = ('FL', 'FR', 'RL', 'RR')):
        """ Constructor of the WBInterface class

        Args:
            initial_feet_pos (LegsAttr): initial feet positions, otherwise they will be all zero
            legs_order (tuple[str, str, str, str], optional): order of the leg. Defaults to ('FL', 'FR', 'RL', 'RR').
        """        
        
        
        mpc_dt = cfg.mpc_params['dt']
        horizon = cfg.mpc_params['horizon']
        self.legs_order = legs_order
        
        # Periodic gait generator --------------------------------------------------------------
        gait_name = cfg.simulation_params['gait']
        gait_params = cfg.simulation_params['gait_params'][gait_name]
        gait_type, duty_factor, step_frequency = gait_params['type'], gait_params['duty_factor'], gait_params['step_freq']
        # Given the possibility to use nonuniform discretization,
        # we generate a contact sequence two times longer and with a dt half of the one of the mpc
        self.pgg = PeriodicGaitGenerator(duty_factor=duty_factor,
                                    step_freq=step_frequency,
                                    gait_type=gait_type,
                                    horizon=horizon)
        # in the case of nonuniform discretization, we create a list of dts and horizons for each nonuniform discretization
        if (cfg.mpc_params['use_nonuniform_discretization']):
            self.contact_sequence_dts = [cfg.mpc_params['dt_fine_grained'], mpc_dt]
            self.contact_sequence_lenghts = [cfg.mpc_params['horizon_fine_grained'], horizon]
        else:
            self.contact_sequence_dts = [mpc_dt]
            self.contact_sequence_lenghts = [horizon]

        
        # Create the foothold reference generator ------------------------------------------------
        stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
        self.frg = FootholdReferenceGenerator(stance_time=stance_time, hip_height=cfg.hip_height, lift_off_positions=initial_feet_pos)


        # Create swing trajectory generator ------------------------------------------------------
        self.step_height = cfg.simulation_params['step_height']
        swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq) 
        position_gain_fb = cfg.simulation_params['swing_position_gain_fb']
        velocity_gain_fb = cfg.simulation_params['swing_velocity_gain_fb']
        swing_generator = cfg.simulation_params['swing_generator']
        self.stc = SwingTrajectoryController(step_height=self.step_height, swing_period=swing_period,
                                        position_gain_fb=position_gain_fb, velocity_gain_fb=velocity_gain_fb,
                                        generator=swing_generator)


        # Terrain estimator -----------------------------------------------------------------------
        self.terrain_computation = TerrainEstimator()

        if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
            # Visual foothold adaptation -------------------------------------------------------------
            self.vfa = VisualFootholdAdaptation(legs_order=self.legs_order, adaptation_strategy=cfg.simulation_params['visual_foothold_adaptation'])


        self.current_contact = np.array([1, 1, 1, 1])


    
    def update_state_and_reference(self,
                                   base_pos: np.ndarray,
                                   base_lin_vel: np.ndarray,
                                   base_ori_euler_xyz: np.ndarray,
                                   base_ang_vel: np.ndarray,
                                   feet_pos: LegsAttr,
                                   hip_pos: LegsAttr,
                                   joints_pos: LegsAttr,
                                   heightmaps,
                                   legs_order: tuple[str, str, str, str],
                                   simulation_dt: float,
                                   ref_base_lin_vel: np.ndarray,
                                   ref_base_ang_vel: np.ndarray) -> [dict, dict, list, LegsAttr, list, list, float, bool]:
        """Update the state and reference for the whole body controller, including the contact sequence, footholds, and terrain estimation.

        Args:
            base_pos (np.ndarray): base position in world frame
            base_lin_vel (np.ndarray): base linear velocity in world frame
            base_ori_euler_xyz (np.ndarray): base orientation in euler angles in world frame
            base_ang_vel (np.ndarray): base angular velocity in base frame
            feet_pos (LegsAttr): feet positions in world frame
            hip_pos (LegsAttr): hip positions in world frame
            joints_pos (LegsAttr): joint positions
            heightmaps (dict): heightmaps for each leg
            legs_order (tuple[str, str, str, str]): order of the legs
            simulation_dt (float): simulation time step
            ref_base_lin_vel (np.ndarray): reference base linear velocity in world frame
            ref_base_ang_vel (np.ndarray): reference base angular velocity in world frame

        Returns:
            state_current (dict): dictionary of the state of the robot that is used in the mpc
            ref_state (dict):  dictionary of the reference state of the robot that is used in the mpc
            contact_sequence (np.ndarray): this is an array, containing the contact sequence of the robot in the future
            ref_feet_pos (LegsAttr): where to step in world frame
            ref_feet_constraints (LegsAttr): constraints for the footholds in the world frame
            contact_sequence_dts (list): 
            contact_sequence_lenghts (list): 
            step_height (float): step height
            optimize_swing (bool), boolean to inform that the robot is in the apex, hence we can optimize step freq. 
        """
        
        state_current = dict(
            position=base_pos,
            linear_velocity=base_lin_vel,
            orientation=base_ori_euler_xyz,
            angular_velocity=base_ang_vel,
            foot_FL=feet_pos.FL,
            foot_FR=feet_pos.FR,
            foot_RL=feet_pos.RL,
            foot_RR=feet_pos.RR,
            joint_FL=joints_pos.FL,
            joint_FR=joints_pos.FR,
            joint_RL=joints_pos.RL,
            joint_RR=joints_pos.RR,
            )


        # Update the desired contact sequence ---------------------------
        self.pgg.run(simulation_dt, self.pgg.step_freq)
        contact_sequence = self.pgg.compute_contact_sequence(contact_sequence_dts=self.contact_sequence_dts, 
                                                contact_sequence_lenghts=self.contact_sequence_lenghts)


        previous_contact = self.current_contact
        self.current_contact = np.array([contact_sequence[0][0],
                                    contact_sequence[1][0],
                                    contact_sequence[2][0],
                                    contact_sequence[3][0]])


        # Compute the reference for the footholds ---------------------------------------------------
        self.frg.update_lift_off_positions(previous_contact, self.current_contact, feet_pos, legs_order)
        ref_feet_pos = self.frg.compute_footholds_reference(
            com_position=base_pos,
            base_ori_euler_xyz=base_ori_euler_xyz,
            base_xy_lin_vel=base_lin_vel[0:2],
            ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
            hips_position=hip_pos,
            com_height_nominal=cfg.simulation_params['ref_z'])
        

        # Adjust the footholds given the terrain -----------------------------------------------------
        if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
            
            time_adaptation = time.time()
            if(self.stc.check_apex_condition(self.current_contact, interval=0.01) and self.vfa.initialized == False):
                for leg_id, leg_name in enumerate(legs_order):
                    heightmaps[leg_name].update_height_map(ref_feet_pos[leg_name], yaw=base_ori_euler_xyz[2])
                self.vfa.compute_adaptation(legs_order, ref_feet_pos, hip_pos, heightmaps, base_lin_vel, base_ori_euler_xyz, base_ang_vel)
                #print("Adaptation time: ", time.time() - time_adaptation)
            
            if(self.stc.check_full_stance_condition(self.current_contact)):
                self.vfa.reset()
            
            ref_feet_pos, ref_feet_constraints = self.vfa.get_footholds_adapted(ref_feet_pos)


            """step_height_modification = False
            for _, leg_name in enumerate(legs_order):
                #TODO this should be in the terrain frame
                ref_feet_pos_height_diffence = ref_feet_pos[leg_name][2] - self.frg.lift_off_positions[leg_name][2]
                if(ref_feet_pos_height_diffence > (cfg.simulation_params['step_height']) and ref_feet_pos_height_diffence > 0.0):
                    step_height = ref_feet_pos_height_diffence + 0.05
                    step_height_modification = True
                    self.stc.regenerate_swing_trajectory_generator(step_height=step_height, swing_period=self.stc.swing_period)
            if(step_height_modification == False):
                step_height = cfg.simulation_params['step_height']
                self.stc.regenerate_swing_trajectory_generator(step_height=step_height, swing_period=self.stc.swing_period)"""
        
        else:
            ref_feet_constraints = LegsAttr(FL=None, FR=None, RL=None, RR=None)
        

        # Estimate the terrain slope and elevation -------------------------------------------------------
        terrain_roll, \
            terrain_pitch, \
            terrain_height = self.terrain_computation.compute_terrain_estimation(
            base_position=base_pos,
            yaw=base_ori_euler_xyz[2],
            feet_pos=self.frg.lift_off_positions,
            current_contact=self.current_contact)

        ref_pos = np.array([0, 0, cfg.hip_height])
        ref_pos[2] = cfg.simulation_params['ref_z'] + terrain_height


        # Update state reference ------------------------------------------------------------------------
        if(cfg.mpc_params['type'] != 'kinodynamic'):
            ref_state = {}
            ref_state |= dict(ref_foot_FL=ref_feet_pos.FL.reshape((1, 3)),
                                ref_foot_FR=ref_feet_pos.FR.reshape((1, 3)),
                                ref_foot_RL=ref_feet_pos.RL.reshape((1, 3)),
                                ref_foot_RR=ref_feet_pos.RR.reshape((1, 3)),
                                # Also update the reference base linear velocity and
                                ref_linear_velocity=ref_base_lin_vel,
                                ref_angular_velocity=ref_base_ang_vel,
                                ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                                ref_position=ref_pos
                                )
        else:
            # In the case of the kinodynamic model,
            # we should pass as a reference the X-Y-Z spline of the feet for the horizon, 
            # since in the kynodimic model we are using the feet position as a reference
            desired_foot_position_FL = np.zeros((cfg.mpc_params['horizon'], 3))
            desired_foot_position_FR = np.zeros((cfg.mpc_params['horizon'], 3))
            desired_foot_position_RL = np.zeros((cfg.mpc_params['horizon'], 3))
            desired_foot_position_RR = np.zeros((cfg.mpc_params['horizon'], 3))
            for leg_id, leg_name in enumerate(legs_order):
                lifted_off = [False, False, False, False]
                for n in range(cfg.mpc_params['horizon']):
                    dt_increment_swing = (n)*cfg.mpc_params['dt']
                    
                    if(lifted_off[leg_id] == False and n >= 0):
                        if(contact_sequence[leg_id][n-1] == 1 and contact_sequence[leg_id][n] == 0):
                            lifted_off[leg_id] = True 

                    if(leg_id == 0):
                       
                        if(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == False):
                            desired_foot_position_FL[n] = feet_pos.FL
                        elif(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == True):
                            desired_foot_position_FL[n] = ref_feet_pos.FL
                        else:
                            desired_foot_position, \
                            desired_foot_velocity, \
                            _ = self.stc.swing_generator.compute_trajectory_references(self.stc.swing_time[leg_id] + dt_increment_swing, 
                                                                                  self.frg.lift_off_positions[leg_name], 
                                                                                  ref_feet_pos.FL)
                            desired_foot_position_FL[n] = desired_foot_position

                    elif(leg_id == 1):
                     
                        if(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == False):
                            desired_foot_position_FR[n] = feet_pos.FR
                        elif(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == True):
                            desired_foot_position_FR[n] = ref_feet_pos.FR
                        else:
                            desired_foot_position, \
                            desired_foot_velocity, \
                            _ = self.stc.swing_generator.compute_trajectory_references(self.stc.swing_time[leg_id]  + dt_increment_swing, 
                                                                                  self.frg.lift_off_positions[leg_name], 
                                                                                  ref_feet_pos.FR)
                            desired_foot_position_FR[n] = desired_foot_position

                    elif(leg_id == 2):
                       
                        if(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == False):
                            desired_foot_position_RL[n] = feet_pos.RL
                        elif(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == True):
                            desired_foot_position_RL[n] = ref_feet_pos.RL
                        else:
                            desired_foot_position, \
                            desired_foot_velocity, \
                            _ = self.stc.swing_generator.compute_trajectory_references(self.stc.swing_time[leg_id]  + dt_increment_swing, 
                                                                                  self.frg.lift_off_positions[leg_name], 
                                                                                  ref_feet_pos.RL)
                            desired_foot_position_RL[n] = desired_foot_position

                    elif(leg_id == 3):
                      
                        if(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == False):
                            desired_foot_position_RR[n] = feet_pos.RR
                        elif(contact_sequence[leg_id][n] == 1 and lifted_off[leg_id] == True):
                            desired_foot_position_RR[n] = ref_feet_pos.RR
                        else:
                            desired_foot_position, \
                            desired_foot_velocity, \
                            _ = self.stc.swing_generator.compute_trajectory_references(self.stc.swing_time[leg_id]  + dt_increment_swing, 
                                                                                  self.frg.lift_off_positions[leg_name], 
                                                                                  ref_feet_pos.RR)
                            desired_foot_position_RR[n] = desired_foot_position
            
            #TODO make this more general
            ref_state = {}
            init_qpos = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
            ref_state |= dict(ref_foot_FL=desired_foot_position_FL,
                                ref_foot_FR=desired_foot_position_FR,
                                ref_foot_RL=desired_foot_position_RL,
                                ref_foot_RR=desired_foot_position_RR,
                                # Also update the reference base linear velocity and
                                ref_linear_velocity=ref_base_lin_vel,
                                ref_angular_velocity=ref_base_ang_vel,
                                ref_orientation=np.array([terrain_roll, terrain_pitch, 0.0]),
                                ref_position=ref_pos,
                                ref_joints=init_qpos
                                )

    
        # -------------------------------------------------------------------------------------------------


        if(cfg.mpc_params['optimize_step_freq']):
            # we can always optimize the step freq, or just at the apex of the swing
            # to avoid possible jittering in the solution
            optimize_swing = self.stc.check_apex_condition(self.current_contact)
        else:
            optimize_swing = 0

        return state_current, ref_state, contact_sequence, ref_feet_pos, ref_feet_constraints, self.contact_sequence_dts, self.contact_sequence_lenghts, self.step_height, optimize_swing
    


    def compute_stance_and_swing_torque(self,
                                        simulation_dt: float,
                                        qpos: np.ndarray,
                                        qvel: np.ndarray,
                                        feet_jac: LegsAttr,
                                        jac_feet_dot: LegsAttr,
                                        feet_pos: LegsAttr,
                                        feet_vel: LegsAttr,
                                        legs_qfrc_bias: LegsAttr,
                                        legs_mass_matrix: LegsAttr,
                                        nmpc_GRFs: LegsAttr,
                                        nmpc_footholds: LegsAttr,
                                        legs_qpos_idx: LegsAttr,
                                        legs_qvel_idx: LegsAttr,
                                        tau: LegsAttr,
                                        optimize_swing: int,
                                        best_sample_freq: float,
                                        nmpc_joints_pos,
                                        nmpc_joints_vel,
                                        nmpc_joints_acc) -> LegsAttr:
        """Compute the stance and swing torque.

        Args:
            simulation_dt (float): simulation time step
            qvel (np.ndarray): joint velocities
            feet_jac (LegsAttr): feet jacobian
            jac_feet_dot (LegsAttr): jacobian of the feet derivative
            feet_pos (LegsAttr): feet positions in world frame
            feet_vel (LegsAttr): feet velocities in world frame
            legs_qfrc_bias (LegsAttr): joint forces and torques
            legs_mass_matrix (LegsAttr): mass matrix of the legs
            nmpc_GRFs (LegsAttr): ground reaction forces from the MPC in world frame 
            nmpc_footholds (LegsAttr): footholds from the MPC in world frame 
            legs_qvel_idx (LegsAttr): joint velocities index
            tau (LegsAttr): joint torques
            optimize_swing (int): flag to signal that we need to update the swing trajectory time
            best_sample_freq (float): best sample frequency obtained from the 
                                      sampling optimization or the batched ocp

        Returns:
            LegsAttr: joint torques
        """
        

        # If we have optimized the gait, we set all the timing parameters
        if (optimize_swing == 1):
            self.pgg.step_freq = np.array([best_sample_freq])[0]
            nominal_sample_freq = self.pgg.step_freq
            self.frg.stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
            swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
            self.stc.regenerate_swing_trajectory_generator(step_height=self.step_height, swing_period=swing_period)
        
        
        # Compute Stance Torque ---------------------------------------------------------------------------
        tau.FL = -np.matmul(feet_jac.FL[:, legs_qvel_idx.FL].T, nmpc_GRFs.FL)
        tau.FR = -np.matmul(feet_jac.FR[:, legs_qvel_idx.FR].T, nmpc_GRFs.FR)
        tau.RL = -np.matmul(feet_jac.RL[:, legs_qvel_idx.RL].T, nmpc_GRFs.RL)
        tau.RR = -np.matmul(feet_jac.RR[:, legs_qvel_idx.RR].T, nmpc_GRFs.RR)



        self.stc.update_swing_time(self.current_contact, self.legs_order, simulation_dt)


        # Compute Swing Torque ------------------------------------------------------------------------------
        if(cfg.mpc_params['type'] != 'kinodynamic'):
            # The swing controller is in the end-effector space. For its computation,
            # we save for simplicity joints position and velocities
            for leg_id, leg_name in enumerate(self.legs_order):
                if self.current_contact[leg_id] == 0:  # If in swing phase, compute the swing trajectory tracking control.
                    tau[leg_name], _, _ = self.stc.compute_swing_control(
                        leg_id=leg_id,
                        q_dot=qvel[legs_qvel_idx[leg_name]],
                        J=feet_jac[leg_name][:, legs_qvel_idx[leg_name]],
                        J_dot=jac_feet_dot[leg_name][:, legs_qvel_idx[leg_name]],
                        lift_off=self.frg.lift_off_positions[leg_name],
                        touch_down=nmpc_footholds[leg_name],
                        foot_pos=feet_pos[leg_name],
                        foot_vel=feet_vel[leg_name],
                        h=legs_qfrc_bias[leg_name],
                        mass_matrix=legs_mass_matrix[leg_name]
                        )
        else:
            for leg_id, leg_name in enumerate(self.legs_order):
                if self.current_contact[leg_id] == 0:
                    error_position = nmpc_joints_pos[leg_name] - qpos[legs_qpos_idx[leg_name]]
                    error_position = error_position.reshape((3, ))

                    error_velocity = nmpc_joints_vel[leg_name] - qvel[legs_qvel_idx[leg_name]]
                    error_velocity = error_velocity.reshape((3, ))

                    accelleration = nmpc_joints_acc[leg_name] 
                    accelleration = accelleration.reshape((3,))
                    
                    # Feedback linearization
                    tau_pd = legs_mass_matrix[leg_name]@(accelleration + 100*error_position + 10*error_velocity) + legs_qfrc_bias[leg_name]
                    tau_pd += 100*error_position + 10*error_velocity
                    tau[leg_name] += tau_pd
                
        return tau
    


    def reset(self, 
              initial_feet_pos: LegsAttr):
        """Reset the whole body interface

        Args:
            initial_feet_pos (LegsAttr): initial feet positions
        """
        
        self.pgg.reset()
        #self.frg.reset()
        #self.stc.reset()
        #self.terrain_computation.reset()
        self.frg.lift_off_positions = initial_feet_pos
        if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
            self.vfa.reset()
        self.current_contact = np.array([1, 1, 1, 1])
        return