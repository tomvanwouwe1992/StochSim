
import numpy as np
import casadi as ca
# Forward dynamics of the 2D arm model - as in Li & Todorov's paper: Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems.
def armForwardDynamics(tau_bio,tau_ext,q,qdot,auxdata):
    a1 = auxdata['I1'] + auxdata['I2'] + auxdata['m2'] * (auxdata['l1'] ** 2)
    a2 = auxdata['m2'] * auxdata['l1'] * auxdata['lc2']
    a3 = auxdata['I2']
    # M = ca.SX.sym('M',2,2)
    # C = ca.SX.sym('C',2,1)
    # B = ca.SX.sym('B',2,2)
    # K = ca.SX.sym('K',2,1)
    M = np.zeros((2,2))
    C = np.zeros((2,))
    B = np.zeros((2, 2))
    M[0, 0] = a1 + 2 * a2 * np.cos(q[1])
    M[0, 1] = a3 + a2 * np.cos(q[1])
    M[1, 0] = a3 + a2 * np.cos(q[1])
    M[1, 1] = a3
    M_inv =  np.linalg.inv(M)
    C[0] = -qdot[1] * (2 * qdot[0] + qdot[1])
    C[1] = qdot[0] ** 2
    C = C * a2 * np.sin(q[1])
    B[0, 0] = 0.05
    B[0, 1] = 0.025
    B[1, 0] = 0.025
    B[1, 1] = 0.05
    qdotdot = np.matmul( M_inv, (tau_bio + tau_ext - C - np.matmul(B, qdot)))
    # ca.mtimes(M_inv, (tau_bio + tau_ext - C - ca.mtimes(B , qdot)))

    return qdotdot

def armForwardDynamics_cas(tau_bio,tau_ext,q,qdot,auxdata):
    a1 = auxdata['I1'] + auxdata['I2'] + auxdata['m2'] * (auxdata['l1'] ** 2)
    a2 = auxdata['m2'] * auxdata['l1'] * auxdata['lc2']
    a3 = auxdata['I2']
    M = ca.SX.sym('M',2,2)
    C = ca.SX.sym('C',2,1)
    B = ca.SX.sym('B',2,2)
    M[0, 0] = a1 + 2 * a2 * np.cos(q[1])
    M[0, 1] = a3 + a2 * np.cos(q[1])
    M[1, 0] = a3 + a2 * np.cos(q[1])
    M[1, 1] = a3
    M_inv =  ca.inv(M)
    C[0] = -qdot[1] * (2 * qdot[0] + qdot[1])
    C[1] = qdot[0] ** 2
    C = C * a2 * np.sin(q[1])
    B[0, 0] = 0.05
    B[0, 1] = 0.025
    B[1, 0] = 0.025
    B[1, 1] = 0.05
    qdotdot = ca.mtimes(M_inv, (tau_bio + tau_ext - C - ca.mtimes(B , qdot)))

    return qdotdot

def handPosition(q, auxdata):
    hand_position = ca.SX.sym('hand_position',2,1)
    hand_position[0,0] = np.cos(q[0]) * auxdata['l1'] + np.cos(q[0] + q[1]) * auxdata['l2']
    hand_position[1,0] = np.sin(q[0]) * auxdata['l1'] + np.sin(q[0] + q[1]) * auxdata['l2']
    return hand_position

def handVelocity(q, qdot, auxdata):
    hand_velocity = ca.SX.sym('hand_velocity', 2, 1)
    a = q[0] + q[1]
    da = qdot[0] + qdot[1]
    hand_velocity[0,0] = qdot[0] * np.sin(q[0]) * auxdata['l1'] + da * np.sin(a) * auxdata['l2']
    hand_velocity[1,0] = -qdot[0] * np.cos(q[0]) * auxdata['l1'] + da * np.cos(a) * auxdata['l2']
    return hand_velocity




