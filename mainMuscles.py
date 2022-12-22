import numpy as np
from ArmModel import armForwardDynamics, handPosition, handVelocity
from MusculoskeletalModel import musculoskeletalDynamics, musculoskeletalGeometryParameters
from Integrator import G_Euler_Explicit
from auxFunctions import LvectoLmat, LvectoLmat_10
import matplotlib.pyplot as plt
import casadi as ca


# Flags
run_shooting_OCP = False
forward_sim_UT = True
run_shooting_UT_OCP = True


nTOT = 10
kappa = 0
alpha = 1
beta = 2
lambda_ = alpha ** 2 * (nTOT + kappa) - nTOT

W_0_M = lambda_ / (nTOT+ lambda_ )
W_i_M = 1 / (2 * (nTOT+ lambda_ ))
W_0_C = lambda_ / (nTOT+ lambda_ ) + 1 - alpha ** 2 + beta
W_i_C = 1 / (2 * (nTOT+ lambda_ ))

WeightVec_M = np.concatenate( (np.reshape(W_0_M,(1,1)), np.tile(W_i_M,(1,2 * nTOT))),1)
WeightVec_C = np.concatenate( (np.reshape(W_0_C,(1,1)), np.tile(W_i_C,(1,2 * nTOT))),1)
c = np.sqrt(nTOT + lambda_ )

# Create dict with model parameters
auxdata = {}
auxdata['l1'] = 0.3
auxdata['l2'] = 0.33
auxdata['lc1'] = 0.11
auxdata['lc2'] = 0.16
auxdata['m1'] = 1.4
auxdata['m2'] = 1
auxdata['I1'] = 0.025
auxdata['I2'] = 0.045
auxdata['muscleDampingCoefficient'] = 0.01
auxdata['FMo'] = 31.8 * np.array((18, 14, 22, 12, 5, 10))
auxdata['tau'] = 0.05
auxdata = musculoskeletalGeometryParameters(auxdata)

exc_SX = ca.SX.sym('exc_SX', 6, 1)
X_SX = ca.SX.sym('X_SX', 10, 1)
q_SX = ca.SX.sym('q_SX', 2, 1)
qdot_SX = ca.SX.sym('qdot_SX', 2, 1)

f_musculoskeletalDynamics = ca.Function('f_musculoskeletalDynamics', [X_SX, exc_SX],
                                        [musculoskeletalDynamics(X_SX, exc_SX, auxdata)])
f_handPosition = ca.Function('f_handPosition', [q_SX], [handPosition(q_SX, auxdata)])
f_dhandPosition_dq = ca.Function('f_dhandPosition_dq', [q_SX], [ca.jacobian(f_handPosition(q_SX), q_SX)])
f_handVelocity = ca.Function('f_handVelocity', [q_SX, qdot_SX], [handVelocity(q_SX, qdot_SX, auxdata)])

finalTime = 1
dt = 0.01
N = int(finalTime / dt)
N_cov = int(N/10)
N_shoot_cov = int(N / N_cov)

# # # Run forward simulation with solution
# X_traj = np.zeros((10, N + 1))
#
# X_traj[6,0] = np.pi * 45 / 180
# X_traj[7,0] = np.pi * 90 / 180
#
# for i in range(N):
#     X = np.reshape(X_traj[:, i], (10, 1))
#     dX = f_musculoskeletalDynamics(X, np.zeros((6,1))).full()
#
#     X_next = G_Euler_Explicit(X, dX, dt)
#     X_traj[:, i + 1] = np.reshape(X_next,(10,))
#
# plt.plot(180 / np.pi * X_traj[6, :])
# plt.plot(180 / np.pi * X_traj[7, :])
# handPos_det = f_handPosition(X_traj[6:8, -1])



if run_shooting_OCP == True:

    precision = 0.001 # 1cm
    target = np.array((0,0.52)) # in Cartesian coordinates
    initial_q = np.pi / 180 * np.array((20, 130)) # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0)) # degrees to radian
    initial_a = np.zeros((6,1))
    endpoint_velocity = 0.001

    initial_hand_position = f_handPosition(initial_q)
    P_init = np.diag(0.01*np.ones((4)))


    opti = ca.Opti()
    exc = opti.variable(6,N)
    opti.subject_to(opti.bounded(0, exc, 1))
    a = initial_a
    q = initial_q
    qdot = initial_qdot
    X = ca.vertcat(a, q, qdot)

    J = 0
    for j in range(N_cov):
        exc_shoot_cov = exc[:,j * N_shoot_cov : (j + 1) * N_shoot_cov]
        for i in range(N_shoot_cov):
            dX = f_musculoskeletalDynamics(X, exc_shoot_cov[:,i])
            X_next = G_Euler_Explicit(X, dX, dt)
            a = X_next[:6]
            q = X_next[6:8]
            qdot = X_next[8:]
            X = X_next
            J = J + exc_shoot_cov[0,i] ** 2 + exc_shoot_cov[1,i] ** 2 + exc_shoot_cov[2,i] ** 2 + exc_shoot_cov[3,i] ** 2 + exc_shoot_cov[4,i] ** 2 + exc_shoot_cov[5,i] ** 2

    final_hand_position = f_handPosition(q)
    final_hand_velocity = f_handVelocity(q, qdot)
    opti.subject_to(final_hand_position > target - precision)
    opti.subject_to(final_hand_position < target + precision)
    # opti.subject_to(final_hand_velocity > - endpoint_velocity)
    # opti.subject_to(final_hand_velocity < + endpoint_velocity)
    opti.minimize(J)

    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 10000,
              "tol": 10 ** (-4)}
    p_opts = {"expand": False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    exc_sol = sol.value(exc)
    plt.plot(exc_sol.T)
    plt.show()


    # Run forward simulation with solution
    # q_traj = np.zeros((2, N + 1))
    # q_traj[:, 0] = initial_q
    # qdot_traj = np.zeros((2, N + 1))
    #
    # for i in range(N):
    #     qdotdot = f_armForwardDynamics(tau_biological_sol[:,i], np.zeros((2,)), q_traj[:, i], qdot_traj[:, i]).full()
    #     X = np.concatenate((q_traj[:, i], qdot_traj[:, i]))
    #     dX = np.concatenate((qdot_traj[:, i], np.reshape(qdotdot,(2,))))
    #     X_next = G_Euler_Explicit(X, dX, dt)
    #     q_traj[:, i + 1] = X_next[:2]
    #     qdot_traj[:, i + 1] = X_next[2:]
    #
    # plt.plot(q_traj[0, :])
    # plt.plot(q_traj[1, :])
    # handPos_det = f_handPosition(q_traj[:2, -1])
    # # plt.show()

#######################################################################################################################

# if forward_sim_UT == True and run_shooting_OCP == True:
#
#     finalTime = 1
#     dt = 0.01
#     N = int(finalTime / dt)
#     N_cov = int(N / 10)
#     N_shoot_cov = int(N / N_cov)
#
#     precision = 0.001  # 1cm
#     target = np.array((0, 0.52))  # in Cartesian coordinates
#     initial_q = np.pi / 180 * np.array((20, 130))  # degrees to radian
#     initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
#     initial_a = np.zeros((6, 1))
#     endpoint_velocity = 0.001
#
#     P_init = np.diag(0.001 * np.ones((10)))
#
#     q = initial_q
#     qdot = initial_qdot
#
#     L_j = ca.chol(P_init)
#     L_mat = LvectoLmat(L_j)
#
#     for j in range(N_cov):
#         A = c * L_mat
#         X = ca.vertcat(q, qdot)
#
#         Y = ca.horzcat(X, X, X, X)
#         sigma = ca.horzcat(X, Y + A, Y - A)
#
#         exc_shoot_cov = exc_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
#
#         for k in range(2 * nTOT + 1):
#             q = sigma[:2, k]
#             qdot = sigma[2:, k]
#
#             for i in range(N_shoot_cov):
#                 dX = f_musculoskeletalDynamics(X, exc_shoot_cov[:, i])
#                 X_next = G_Euler_Explicit(X, dX, dt)
#                 a = X_next[:6]
#                 q = X_next[6:8]
#                 qdot = X_next[8:]
#                 X = X_next
#                 J = J + exc_shoot_cov[0, i] ** 2 + exc_shoot_cov[1, i] ** 2 + exc_shoot_cov[2, i] ** 2 + exc_shoot_cov[
#                     3, i] ** 2 + exc_shoot_cov[4, i] ** 2 + exc_shoot_cov[5, i] ** 2
#
#
#                 qdotdot = f_armForwardDynamics(tau_biological_shoot_cov[:, i], np.zeros((2,)), q, qdot)
#                 X = ca.vertcat(q, qdot)
#                 dX = ca.vertcat(qdot, qdotdot)
#                 X_next = G_Euler_Explicit(X, dX, dt)
#                 q = X_next[:2]
#                 qdot = X_next[2:]
#
#             if k == 0:
#                 sigma_next = X_next
#                 J = J + tau_biological_shoot_cov[0, i] ** 2 + tau_biological_shoot_cov[1, i] ** 2
#             else:
#                 sigma_next = ca.horzcat(sigma_next, X_next)
#
#         mean_next = ca.mtimes(sigma_next, WeightVec_M.T)
#         Y1k = sigma_next - mean_next
#         P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
#         L_mat = ca.chol(P_mat_next)
#         print(mean_next)
#
#     print(L_mat)
#     print(sigma_next)
#     print(mean_next)

tau_bio_SX = ca.SX.sym('exc_SX', 2, 1)
tau_ext_SX = ca.SX.sym('tau_ext_SX', 2, 1)
q_SX = ca.SX.sym('q_SX', 2, 1)
qdot_SX = ca.SX.sym('qdot_SX', 2, 1)
L_SX = ca.SX.sym('L_SX', 55, 1)
P_SX = ca.SX.sym('P_SX', 10, 10)
f_armForwardDynamics = ca.Function('f_armForwardDynamics', [tau_bio_SX, tau_ext_SX, q_SX, qdot_SX],
                                   [armForwardDynamics(tau_bio_SX, tau_ext_SX, q_SX, qdot_SX, auxdata)])
f_handPosition = ca.Function('f_handPosition', [q_SX], [handPosition(q_SX, auxdata)])
f_handVelocity = ca.Function('f_handVelocity', [q_SX, qdot_SX], [handVelocity(q_SX, qdot_SX, auxdata)])
f_LvectoLmat = ca.Function('f_LvectoLmat', [L_SX], [LvectoLmat_10(L_SX)])
f_chol = ca.Function('f_chol', [P_SX], [ca.chol(P_SX)])


# Run forward simulation with solution
if forward_sim_UT == True:

    finalTime = 0.5
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N / 10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.00001  # 1cm
    initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
    initial_a = np.zeros((6, 1))
    initial_hand_position = f_handPosition(initial_q)
    target = np.array((0, 0.52))  # in Cartesian coordinates

    endpoint_velocity = 0.001

    P_init = np.diag(0.0001 * np.ones((10)))
    P_full = np.zeros((10, 10, N_cov + 1))
    L_mat = f_chol(P_init)
    P_full[:, :, 0] = P_init

    a = initial_a
    q = initial_q
    qdot = initial_qdot
    mean_next = ca.vertcat(a, q, qdot)

    mean_previous = mean_next
    traj = np.ones((10, N_cov + 1))

    traj[:, 0] = np.reshape(mean_next, (10,))

    exc_sol = np.zeros((6,N+1))
    k_feedback_sol = np.zeros((6, 4 * (N_cov+1)))

    for j in range(N_cov):
        A = c * L_mat.T
        X = mean_next

        Y = ca.horzcat(X, X, X, X, X, X, X, X, X, X)
        sigma = ca.horzcat(X, Y + A, Y - A)
        exc_cov = exc_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
        k_feedback_shoot_cov = k_feedback_sol[:, j * 4: (j + 1) * 4]

        # We integrate every sigma point seperately within one multiple shooting interval
        for k in range(2 * nTOT + 1):
            a = sigma[:6, k]
            q = sigma[6:8, k]
            qdot = sigma[8:, k]
            X = ca.vertcat(a, q, qdot)

            # Feedback torque stays constant over on multiple shooting interval (mean_previous = delay on feedback torque)
            exc_feedback = ca.mtimes(k_feedback_shoot_cov, X[6:] - mean_previous[6:])

            # Forward integration from the start to the end of one multiple shooting interval
            for i in range(N_shoot_cov):
                dX = f_musculoskeletalDynamics(X, exc_cov[:, i] + exc_feedback)
                X_next = G_Euler_Explicit(X, dX, dt)
                a = X_next[:6]
                q = X_next[6:8]
                qdot = X_next[8:]
                X = X_next
            # Collect sigma points at the end of the interval
            if k == 0:
                sigma_next = X_next
            else:
                sigma_next = ca.horzcat(sigma_next, X_next)

        mean_previous = mean_next
        mean_next = ca.mtimes(sigma_next, WeightVec_M.T)
        Y1k = sigma_next - mean_next
        P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
        P_full[:, :, j + 1] = P_mat_next
        L_mat = f_chol(P_mat_next)

print(P_full)



#######################################################################################################################

if run_shooting_UT_OCP == True:
    # The goal is to reach a target within a chose precision in a specified time with minimal effort starting from a specific pose
    finalTime = 0.5
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N / 10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.00001  # 1cm
    initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
    initial_a = np.zeros((6,1))
    initial_hand_position = f_handPosition(initial_q)
    target = np.array((0, 0.52))  # in Cartesian coordinates

    endpoint_velocity = 0.001

    P_init = np.diag(0.0001 * np.ones((10)))

    opti_robust = ca.Opti()
    exc = opti_robust.variable(6, N)
    opti_robust.subject_to(opti_robust.bounded(0, exc, 1))
    k_feedback = opti_robust.variable(6, 4 * N_cov)

    opti_robust.set_initial(exc,0.1)
    sigma_next = ca.SX.sym('sigma_next', nTOT, 2 * nTOT + 1)
    q = initial_q
    qdot = initial_qdot
    a = np.zeros((6,1))

    mean = ca.vertcat(a, q, qdot)
    mean_previous = mean
    mean_previous_previous = mean
    J = 0
    L_mat = f_chol(P_init)

    # Two loops of forward simulation (~ multiple shooting)
    # Outer loop - number of multiple shooting interval
    for j in range(N_cov):
        A = c * L_mat.T
        X = mean

        Y = ca.horzcat(X,X,X,X,X,X,X,X,X,X)
        sigma = ca.horzcat(X, Y + A, Y - A)
        exc_cov = exc[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
        k_feedback_shoot_cov = k_feedback[:, j * 4: (j + 1) * 4]
        if j == 0:
            sigma_previous = sigma

        # We integrate every sigma point seperately within one multiple shooting interval (this would be good to parallelize!)
        for k in range(2 * nTOT + 1):
            a = sigma[:6, k]
            q = sigma[6:8, k]
            qdot = sigma[8:, k]
            X = ca.vertcat(a, q, qdot)
            a_previous = sigma_previous[:6, k]
            q_previous = sigma_previous[6:8, k]
            qdot_previous = sigma_previous[8:, k]
            if j == 0:
                X_previous_previous = ca.vertcat(a_previous, q_previous, qdot_previous)
            else:
                X_previous_previous = X_previous
            X_previous = ca.vertcat(a_previous, q_previous, qdot_previous)

            X_delayed = X_previous
            mean_delayed = mean_previous
            # Feedback torque stays constant over on multiple shooting interval (mean_previous = delay on feedback torque)
            exc_feedback = ca.mtimes(k_feedback_shoot_cov, X_delayed[6:] - mean_delayed[6:])


            # Forward integration from the start to the end of one multiple shooting interval
            for i in range(N_shoot_cov):
                dX = f_musculoskeletalDynamics(X, exc_cov[:, i] + exc_feedback)
                X_next = G_Euler_Explicit(X, dX, dt)
                a = X_next[:6]
                q = X_next[6:8]
                qdot = X_next[8:]
                X = X_next

                # The feedforward and feedback torque squared are added to the cost function
                J = J + (a[0] ** 2 + a[1] ** 2 + a[2] ** 2 + a[3] ** 2 + a[4] ** 2 + a[5] ** 2) / (2 * nTOT + 1)
                J = J + (exc_cov[0, i] ** 2 + exc_cov[1, i] ** 2 + exc_cov[2, i] ** 2 + exc_cov[3, i] ** 2 + exc_cov[4, i] ** 2 + exc_cov[5, i] ** 2) / (2 * nTOT + 1)
                J = J + (exc_feedback[0] ** 2 + exc_feedback[1] ** 2 + exc_feedback[2] ** 2 + exc_feedback[3] ** 2 + exc_feedback[4] ** 2 + exc_feedback[5] ** 2) / (2 * nTOT + 1)
            # Collect sigma points at the end of the interval
            if k == 0:
                sigma_next = X_next
            else:
                sigma_next = ca.horzcat(sigma_next, X_next)

        sigma_previous = sigma
        mean_previous_previous = mean_previous
        mean_previous = mean
        mean = ca.mtimes(sigma_next , WeightVec_M.T)
        Y1k = sigma_next - mean
        P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
        L_mat = f_chol(P_mat_next)

    P_handPosition = f_dhandPosition_dq(mean[6:8])*P_mat_next[6:8,6:8]*f_dhandPosition_dq(mean[6:8]).T

    J = J + (a[0, 0] ** 2 + a[1, 0] ** 2 + a[2, 0] ** 2 + a[3, 0] ** 2 + a[4, 0] ** 2 + a[5, 0] ** 2)
    opti_robust.subject_to(1e3 * (P_handPosition[0, 0] - 0.01 ** 2) < 0)
    opti_robust.subject_to(1e3 * (P_handPosition[1, 1] - 0.01 ** 2) < 0)
    q_final = mean[6:8]
    qdot_final = mean[8:]
    final_hand_position = f_handPosition(q_final)
    final_hand_velocity = f_handVelocity(q_final, qdot_final)
    opti_robust.subject_to(final_hand_position[0] > target[0] - precision)
    opti_robust.subject_to(final_hand_position[1] > target[1] - precision)
    opti_robust.subject_to(final_hand_position[0] < target[0] + precision)
    opti_robust.subject_to(final_hand_position[1] < target[1] + precision)

    opti_robust.minimize(J)

    s_opts = {"linear_solver": "mumps",
              "hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 10000,
              "tol": 10 ** (-1),
              "constr_viol_tol": 10 ** (-4)}
    p_opts = {"expand": False}
    opti_robust.solver("ipopt", p_opts, s_opts)
    sol = opti_robust.solve()

    exc_sol = sol.value(exc)
    a_init_sol = np.zeros((6,1))
    k_feedback_sol = sol.value(k_feedback)
    # L_final_sol = sol.value(L_final)

print(np.mean(exc_sol))
# Run forward simulation with solution
if forward_sim_UT == True:

    finalTime = 0.5
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N / 10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.00001  # 1cm
    initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
    initial_a = a_init_sol
    initial_hand_position = f_handPosition(initial_q)
    target = np.array((0, 0.52))  # in Cartesian coordinates

    endpoint_velocity = 0.001

    P_init = np.diag(0.0001 * np.ones((10)))
    P_full = np.zeros((10, 10, N_cov + 1))
    L_mat = f_chol(P_init)
    P_full[:, :, 0] = P_init

    a = initial_a
    q = initial_q
    qdot = initial_qdot
    mean = ca.vertcat(a, q, qdot)

    mean_previous = mean
    mean_previous_previous = mean
    traj = np.ones((10, N_cov + 1))

    traj[:, 0] = np.reshape(mean, (10,))

    for j in range(N_cov):
        A = c * L_mat.T
        X = mean

        Y = ca.horzcat(X, X, X, X, X, X, X, X, X, X)
        sigma = ca.horzcat(X, Y + A, Y - A)
        exc_cov = exc_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
        k_feedback_shoot_cov = k_feedback_sol[:, j * 4: (j + 1) * 4]
        if j == 0:
            sigma_previous = sigma
        # We integrate every sigma point seperately within one multiple shooting interval
        for k in range(2 * nTOT + 1):
            a = sigma[:6, k]
            q = sigma[6:8, k]
            qdot = sigma[8:, k]
            X = ca.vertcat(a, q, qdot)
            a_previous = sigma_previous[:6, k]
            q_previous = sigma_previous[6:8, k]
            qdot_previous = sigma_previous[8:, k]
            X_previous_previous = X_previous
            X_previous = ca.vertcat(a_previous, q_previous, qdot_previous)

            X_delayed = X_previous
            mean_delayed = mean_previous
            # Feedback torque stays constant over on multiple shooting interval (mean_previous = delay on feedback torque)
            exc_feedback = ca.mtimes(k_feedback_shoot_cov, X_delayed[6:] - mean_delayed[6:])

            # Forward integration from the start to the end of one multiple shooting interval
            for i in range(N_shoot_cov):
                dX = f_musculoskeletalDynamics(X, exc_cov[:, i] + exc_feedback)
                X_next = G_Euler_Explicit(X, dX, dt)
                a = X_next[:6]
                q = X_next[6:8]
                qdot = X_next[8:]
                X = X_next
                hand_position = f_handPosition(q)
            # Collect sigma points at the end of the interval
            if k == 0:
                sigma_next = X_next
            else:
                sigma_next = ca.horzcat(sigma_next, X_next)

        sigma_previous = sigma
        mean_previous_previous = mean_previous
        mean_previous = mean
        mean = ca.mtimes(sigma_next, WeightVec_M.T)
        Y1k = sigma_next - mean
        P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
        P_full[:, :, j + 1] = P_mat_next
        L_mat = f_chol(P_mat_next)

    q_final = mean[6:8]
    final_hand_position = f_handPosition(q_final)
    P_handPosition_final = f_dhandPosition_dq(mean[6:8])*P_full[6:8,6:8,-1]*f_dhandPosition_dq(mean[6:8]).T

    print(P_full)
    print(final_hand_position)






