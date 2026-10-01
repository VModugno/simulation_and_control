import numpy as np

# TODO this method works only for fixed base single kinematic chain robot
def ImpedanceController(dyn_model, controlled_frame_name, q_mes, qd_mes, xdes, kp, kd, tau_ext, pos_or_ori = "pos"):

    # 
    if pos_or_ori == "pos":
        cart_dim = 3
    elif pos_or_ori == "ori":
        cart_dim = 3
    else:
        cart_dim = 6

    # tau_ext preprocessing if it is rotation or position
    if pos_or_ori == "pos":
        tau_ext = tau_ext[:3]
    elif pos_or_ori == "ori":
        tau_ext = tau_ext[3:]
    else:
        tau_ext = tau_ext

    # here i check if kp is a an array or a scalar
    if isinstance(kp, (int, float)):
        P = np.eye(cart_dim) * kp
    else:
        P = np.diag(kp)

    if isinstance(kd, (int, float)):
        D = np.eye(cart_dim) * kd
    else:
        D = np.diag(kd)

    
    dyn_model.ComputeJacobian(q_mes, controlled_frame_name,"local_global") # jacobian
    J = dyn_model.res.J
    # here i split the Jacobian in the linear and angular part
    Jl = J[:3,:] #linear part
    Ja = J[3:,:] #angular part
    if pos_or_ori == "pos":
        J = Jl  
    else:
        J = Ja
          
    xd_ee = J @ qd_mes  #cartesian velocity
    x_ee, _ = dyn_model.ComputeFK(q_mes,controlled_frame_name) # end effector pose
    
    # here i compute the feeback linearization tau // the reordering is already done inside compute all teamrs
    dyn_model.ComputeAllTerms(q_mes, qd_mes)

    M = dyn_model.res.M 
    S = dyn_model.res.c 
    g = dyn_model.res.g    
    
    Mx =  np.linalg.pinv(J).T @ M @ np.linalg.pinv(J) 
    Sx = np.linalg.pinv(J).T @ S                  
    gx = np.linalg.pinv(J).T @ g

    a = np.linalg.inv(Mx) @ (tau_ext -D @ xd_ee + P @ ((xdes - x_ee)))

  
    u = J.T @ (Mx @ a + Sx + gx - tau_ext)
   
    return u