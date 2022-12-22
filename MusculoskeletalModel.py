import numpy as np
import casadi as ca
from ArmModel import armForwardDynamics


def forceToTorque(F, q, auxdata):
    theta_shoulder = q[0]
    theta_elbow = q[1]
    dM_matrix = evaluate_dM_matrix(auxdata['dM_coefficients'],theta_shoulder,theta_elbow)
    T = ca.mtimes(dM_matrix, F)
    return T

def musculoskeletalGeometryParameters(auxdata):
    ## LMT and dM coefficients
    LMT_coefficients = np.zeros((6,2))
    LMT_coefficients[0, 0] = 1.1
    LMT_coefficients[0, 1] = -5.2063
    LMT_coefficients[1, 0] = 0.8
    LMT_coefficients[1, 1] = -7.5389
    LMT_coefficients[2, 0] = 1.2
    LMT_coefficients[2, 1] = -3.9381
    LMT_coefficients[3, 0] = 0.7
    LMT_coefficients[3, 1] = -3.0315
    LMT_coefficients[4, 0] = 1.1
    LMT_coefficients[4, 1] = -2.5228
    LMT_coefficients[5, 0] = 0.85
    LMT_coefficients[5, 1] = -1.8264

    dM_coefficients = np.zeros((6,6))
    dM_coefficients[0, 2] = 0.01
    dM_coefficients[0, 3] = 0.03
    dM_coefficients[0, 4] = -0.011
    dM_coefficients[0, 5] = 1.9

    dM_coefficients[1, 2] = 0.01
    dM_coefficients[1, 3] = -0.019
    dM_coefficients[1, 5] = 0.01

    dM_coefficients[2, 0] = 0.04
    dM_coefficients[2, 1] = -0.008
    dM_coefficients[2, 2] = 1.9
    dM_coefficients[2, 5] = 0.01

    dM_coefficients[3, 0] = -0.042
    dM_coefficients[3, 2] = 0.01
    dM_coefficients[3, 5] = 0.01

    dM_coefficients[4, 0] = 0.03
    dM_coefficients[4, 1] = -0.011
    dM_coefficients[4, 2] = 1.9
    dM_coefficients[4, 3] = 0.032
    dM_coefficients[4, 4] = -0.01
    dM_coefficients[4, 5] = 1.9

    dM_coefficients[5, 0] = -0.039
    dM_coefficients[5, 2] = 0.01
    dM_coefficients[5, 3] = -0.022
    dM_coefficients[5, 5] = 0.01

    auxdata['dM_coefficients'] = dM_coefficients
    auxdata['LMT_coefficients'] = LMT_coefficients

    return auxdata



def getMuscleForce(q,qdot,auxdata):

    ## Muscle tendon unit length and velocity
    lM_tilde = evaluate_LMT_vector(auxdata['dM_coefficients'],auxdata['LMT_coefficients'],q[0],q[1])
    vMtilde = evaluate_VMT_vector(auxdata['dM_coefficients'],auxdata['LMT_coefficients'],q[0],q[1],qdot[0],qdot[1])
    vMtilde_normalizedToMaxVelocity = vMtilde / 10


    ## Compute active muscle force components
    # Coefficients active force-length
    b11 = 0.8145
    b21 = 1.0550
    b31 = 0.1624
    b41 = 0.0633
    b12 = 0.4330
    b22 = 0.7168
    b32 = -0.0299
    b42 = 0.2004
    b13 = 0.1
    b23 = 1
    b33 = 0.5*np.sqrt(0.5)
    b43 = 0

    # Active force-length multiplier is sum of 3 Gaussian functions
    num1 = lM_tilde - b21
    den1 = b31 + b41 * lM_tilde
    FMl_tilde1 = b11 * ca.exp(-0.5 * (num1 ** 2) / (den1 ** 2))

    num2 = lM_tilde - b22
    den2 = b32 + b42 * lM_tilde
    FMl_tilde2 = b12 * ca.exp(-0.5 * (num2 ** 2) / (den2 ** 2))

    num3 = lM_tilde - b23
    den3 = b33 + b43 * lM_tilde
    FMl_tilde3 = b13 * ca.exp(-0.5 * (num3 ** 2) / (den3 ** 2))

    FMl_tilde = FMl_tilde1 + FMl_tilde2 + FMl_tilde3

    # Coefficients force-velocity
    e1 = -0.2158
    e2 = -32.5966
    e3 = -1.1241
    e4 = 0.9126

    # Force-velocity multiplier
    FMv_tilde = e1 * np.log((e2 * vMtilde_normalizedToMaxVelocity + e3) + np.sqrt((e2 * vMtilde_normalizedToMaxVelocity+e3) ** 2 + 1)) + e4

    # Active muscle force multiplier
    Fce = FMl_tilde * FMl_tilde


    ## Passive muscle force component
    # Coefficient passive force length relation
    pas1 = -0.9952
    pas2 = 53.5982
    e0 = 0.6
    kpe = 4
    # Passive force-length: exponential relation
    t5 = np.exp(kpe * (lM_tilde - 1) / e0)
    Fl_pe = ((t5 - 1) - pas1) / pas2
    # Passive force-velocity:damper
    Fv_pe = auxdata['muscleDampingCoefficient'] * vMtilde_normalizedToMaxVelocity

    # Scaled passive and active components
    FMo = auxdata['FMo']
    Fp = FMo * (Fl_pe + Fv_pe)
    Fa = FMo * Fce

    return Fa, Fp

def evaluate_LMT(a_shoulder, b_shoulder, c_shoulder, a_elbow, b_elbow, c_elbow, l_base, l_multiplier, theta_shoulder,
                   theta_elbow):

    l_full = a_shoulder * theta_shoulder + b_shoulder * ca.sin(
        c_shoulder * theta_shoulder) / c_shoulder + a_elbow * theta_elbow + b_elbow * ca.sin(
        c_elbow * theta_elbow) / c_elbow
    LMT = l_full * l_multiplier + l_base

    return LMT

def evaluate_LMT_vector(dM_coefficients, LMT_coefficients, theta_shoulder, theta_elbow):
    LMT_vector = evaluate_LMT(dM_coefficients[:, 0], dM_coefficients[:, 1], dM_coefficients[:, 2], dM_coefficients[:, 3], dM_coefficients[:, 4], dM_coefficients[:, 5], LMT_coefficients[:, 0], LMT_coefficients[:, 1], theta_shoulder, theta_elbow)

    return LMT_vector

def evaluate_dM(a, b, c, theta):
    dM = a + b * ca.cos(c * theta)

    return dM

def evaluate_dM_matrix(dM_coefficients,theta_shoulder,theta_elbow):
    dM_shoulder = evaluate_dM(dM_coefficients[:, 0], dM_coefficients[:, 1], dM_coefficients[:, 2], theta_shoulder)
    dM_elbow =  evaluate_dM(dM_coefficients[:, 3], dM_coefficients[:, 4], dM_coefficients[:, 5], theta_elbow)
    dM_matrix = ca.horzcat(dM_shoulder,dM_elbow)
    dM_matrix = dM_matrix.T

    return dM_matrix


def evaluate_VMT(a_shoulder, b_shoulder, c_shoulder, a_elbow, b_elbow, c_elbow, l_multiplier, theta_shoulder,
                   theta_elbow, dtheta_shoulder, dtheta_elbow):

    # Returns the fiber velocity normalized by optimal fiber length. Units in optimal fiber lengths per second
    v_full = a_shoulder * dtheta_shoulder + b_shoulder * ca.cos(c_shoulder * theta_shoulder) * dtheta_shoulder + a_elbow * dtheta_elbow + b_elbow * ca.cos(c_elbow * theta_elbow) * dtheta_elbow
    VMT = l_multiplier * v_full

    return VMT

def evaluate_VMT_vector(dM_coefficients, LMT_coefficients, theta_shoulder, theta_elbow,dtheta_shoulder,dtheta_elbow):
    VMT_vector = evaluate_VMT(dM_coefficients[:, 0], dM_coefficients[:, 1], dM_coefficients[:, 2], dM_coefficients[:, 3], dM_coefficients[:, 4], dM_coefficients[:, 5], LMT_coefficients[:, 1], theta_shoulder, theta_elbow, dtheta_shoulder, dtheta_elbow)

    return VMT_vector


def musculoskeletalDynamics(X, u, auxdata):
    a = X[:6]
    q = X[6:8]
    qdot = X[8:]

    Fa, Fp = getMuscleForce(q,qdot,auxdata)

    Fm = a * Fa + Fp
    T = forceToTorque(Fm,q,auxdata)
    ddtheta = armForwardDynamics(T, (0,0), q, qdot, auxdata)
    
    dX =  ca.vertcat((u-a) / auxdata['tau'] , qdot, ddtheta)
    
    return dX