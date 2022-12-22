import numpy as np
from ArmModel import armForwardDynamics, handPosition, handVelocity, armForwardDynamics_cas
from Integrator import G_Euler_Explicit
from auxFunctions import LvectoLmat
import matplotlib.pyplot as plt
import casadi as ca


# Flags
run_example_forward = False
run_shooting_OCP = True
forward_sim_UT = False
run_shooting_UT_OCP = True

# Hyperparameters for the unscented transform
nTOT = 4
kappa = 0
alpha = 1
beta = 2
lambda_ = alpha ** 2 * (nTOT + kappa) - nTOT

W_0_M = lambda_ / (nTOT+ lambda_ )
W_i_M = 1 / (2 * (nTOT+ lambda_ ))
W_0_C = lambda_ / (nTOT+ lambda_ )
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

#######################################################################################################################

if run_shooting_OCP == True:

    tau_bio_SX = ca.SX.sym('tau_bio_SX',2,1)
    tau_ext_SX = ca.SX.sym('tau_ext_SX',2,1)
    q_SX = ca.SX.sym('q_SX', 2, 1)
    qdot_SX = ca.SX.sym('qdot_SX', 2, 1)

    f_armForwardDynamics = ca.Function('f_armForwardDynamics', [tau_bio_SX, tau_ext_SX, q_SX, qdot_SX], [armForwardDynamics_cas(tau_bio_SX, tau_ext_SX, q_SX, qdot_SX, auxdata)] )
    f_handPosition = ca.Function('f_handPosition', [q_SX], [handPosition(q_SX,auxdata)])
    f_dhandPosition_dq = ca.Function('f_dhandPosition_dq', [q_SX], [ca.jacobian(f_handPosition(q_SX), q_SX)])
    f_handVelocity = ca.Function('f_handVelocity', [q_SX, qdot_SX], [handVelocity(q_SX,qdot_SX,auxdata)])

    # The goal is to reach a target within a chose precision in a specified time with minimal effort starting from a specific pose
    finalTime = 1
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N/10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.0000001 # 1cm
    target = np.array((0,0.52)) # in Cartesian coordinates
    initial_q = np.pi / 180 * np.array((30, 120)) # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0)) # degrees to radian
    endpoint_velocity = 0.001

    P_init = np.diag(0.01*np.ones((4)))


    opti = ca.Opti()
    tau_biological = opti.variable(2,N)
    opti.set_initial(tau_biological, 0.001)
    q = initial_q
    qdot = initial_qdot
    J = 0
    for j in range(N_cov):
        tau_biological_shoot_cov = tau_biological[:,j * N_shoot_cov : (j + 1) * N_shoot_cov]
        for i in range(N_shoot_cov):
            qdotdot = f_armForwardDynamics(tau_biological_shoot_cov[:,i], np.zeros((2,)), q, qdot)
            X = ca.vertcat(q, qdot)
            dX = ca.vertcat(qdot, qdotdot)
            X_next = G_Euler_Explicit(X, dX, dt)
            q = X_next[:2]
            qdot = X_next[2:]
            J = J + tau_biological_shoot_cov[0,i] ** 2 + tau_biological_shoot_cov[1,i] ** 2

    final_hand_position = f_handPosition(q)
    final_hand_velocity = f_handVelocity(q, qdot)
    opti.subject_to(final_hand_position > target - precision)
    opti.subject_to(final_hand_position < target + precision)
    opti.subject_to(final_hand_velocity > - endpoint_velocity)
    opti.subject_to(final_hand_velocity < + endpoint_velocity)
    opti.minimize(J)

    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 10000,
              "tol": 10 ** (-4)}
    p_opts = {"expand": False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    tau_biological_sol = sol.value(tau_biological)

    # Run forward simulation with solution
    q_traj = np.zeros((2, N + 1))
    q_traj[:, 0] = initial_q
    qdot_traj = np.zeros((2, N + 1))

    for i in range(N):
        qdotdot = f_armForwardDynamics(tau_biological_sol[:,i], np.zeros((2,)), q_traj[:, i], qdot_traj[:, i]).full()
        X = np.concatenate((q_traj[:, i], qdot_traj[:, i]))
        dX = np.concatenate((qdot_traj[:, i], np.reshape(qdotdot,(2,))))
        X_next = G_Euler_Explicit(X, dX, dt)
        q_traj[:, i + 1] = X_next[:2]
        qdot_traj[:, i + 1] = X_next[2:]

    plt.plot(q_traj[0, :])
    plt.plot(q_traj[1, :])
    handPos_det = f_handPosition(q_traj[:2, -1])
    # plt.show()

#######################################################################################################################

if forward_sim_UT == True and run_shooting_OCP == True:

    finalTime = 1
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N / 10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.000001  # 1cm
    target = np.array((0, 0.52))  # in Cartesian coordinates
    initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
    endpoint_velocity = 0.001

    P_init = np.diag(0.01 * np.ones((4)))

    q = initial_q
    qdot = initial_qdot

    L_j = ca.vertcat(0.1,0,0,0,0.1,0,0,0.1,0,0.1)
    L_mat = LvectoLmat(L_j)

    for j in range(N_cov):
        A = c * L_mat
        X = ca.vertcat(q, qdot)

        Y = ca.horzcat(X, X, X, X)
        sigma = ca.horzcat(X, Y + A, Y - A)

        tau_biological_shoot_cov = tau_biological_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]

        for k in range(2 * nTOT + 1):
            q = sigma[:2, k]
            qdot = sigma[2:, k]

            for i in range(N_shoot_cov):
                qdotdot = f_armForwardDynamics(tau_biological_shoot_cov[:, i], np.zeros((2,)), q, qdot)
                X = ca.vertcat(q, qdot)
                dX = ca.vertcat(qdot, qdotdot)
                X_next = G_Euler_Explicit(X, dX, dt)
                q = X_next[:2]
                qdot = X_next[2:]

            if k == 0:
                sigma_next = X_next
                J = J + tau_biological_shoot_cov[0, i] ** 2 + tau_biological_shoot_cov[1, i] ** 2
            else:
                sigma_next = ca.horzcat(sigma_next, X_next)

        mean_next = ca.mtimes(sigma_next, WeightVec_M.T)
        Y1k = sigma_next - mean_next
        P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
        L_mat = ca.chol(P_mat_next)
        print(mean_next)

    print(L_mat)
    print(sigma_next)
    print(mean_next)



#######################################################################################################################

if run_shooting_UT_OCP == True:

    tau_bio_SX = ca.SX.sym('tau_bio_SX', 2, 1)
    tau_ext_SX = ca.SX.sym('tau_ext_SX', 2, 1)
    q_SX = ca.SX.sym('q_SX', 2, 1)
    qdot_SX = ca.SX.sym('qdot_SX', 2, 1)
    L_SX = ca.SX.sym('L_SX', 10, 1)
    P_SX = ca.SX.sym('P_SX', 4, 4)
    f_armForwardDynamics = ca.Function('f_armForwardDynamics', [tau_bio_SX, tau_ext_SX, q_SX, qdot_SX],
                                       [armForwardDynamics_cas(tau_bio_SX, tau_ext_SX, q_SX, qdot_SX, auxdata)])
    f_handPosition = ca.Function('f_handPosition', [q_SX], [handPosition(q_SX, auxdata)])
    f_handVelocity = ca.Function('f_handVelocity', [q_SX, qdot_SX], [handVelocity(q_SX, qdot_SX, auxdata)])
    f_LvectoLmat = ca.Function('f_LvectoLmat', [L_SX], [LvectoLmat(L_SX)])
    f_chol = ca.Function('f_chol', [P_SX], [ca.chol(P_SX)])
    # The goal is to reach a target within a chose precision in a specified time with minimal effort starting from a specific pose
    finalTime = 1
    dt = 0.01
    N = int(finalTime / dt)
    N_cov = int(N / 10)
    N_shoot_cov = int(N / N_cov)

    precision = 0.001  # 1mm
    target = np.array((0, 0.52))  # in Cartesian coordinates
    initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
    initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
    endpoint_velocity = 0.001

    P_init = np.diag(0.01 * np.ones((4)))

    opti = ca.Opti()
    tau_biological = opti.variable(2, N)
    k_feedback = opti.variable(2, 4 * N_cov)
    mean_trajectory = opti.variable(4, N_cov + 1)


    opti.set_initial(tau_biological,0.00001)
    L = opti.variable(10, N_cov + 1)
    opti.set_initial(L , 0.1)
    sigma_next = ca.SX.sym('sigma_next', nTOT, 2 * nTOT + 1)
    q = initial_q
    qdot = initial_qdot
    mean_next = ca.vertcat(q, qdot)
    opti.subject_to(mean_trajectory[:, 0] == mean_next)
    J = 0
    L_j = 0.1*ca.vertcat(0.1,0,0,0,0.1,0,0,0.1,0,0.1)
    # opti.subject_to(L[:,0] - L_j == 0)
    L_mat_next = f_LvectoLmat(L_j)
    P_mat_init = L_mat_next*L_mat_next.T
    # Two loops of forward simulation (~ multiple shooting)
    # Outer loop - number of multiple shooting interval
    for j in range(N_cov):
        A = c * L_mat_next
        X = ca.vertcat(q, qdot)

        Y = ca.horzcat(X,X,X,X)
        sigma = ca.horzcat(X, Y + A, Y - A)
        tau_biological_shoot_cov = tau_biological[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
        k_feedback_shoot_cov = k_feedback[:, j * 4: (j + 1) * 4]
        mean_traj_j = mean_trajectory[:, j]
        # We integrate every sigma point seperately within one multiple shooting interval
        for k in range(2 * nTOT + 1):
            q = sigma[:2, k]
            qdot = sigma[2:, k]
            X = ca.vertcat(q, qdot)

            # Feedback torque stays constant over on multiple shooting interval (mean_previous = delay on feedback torque)
            tau_feedback = ca.mtimes(k_feedback_shoot_cov, X - mean_traj_j)

            # Forward integration from the start to the end of one multiple shooting interval
            for i in range(N_shoot_cov):
                qdotdot = f_armForwardDynamics(tau_biological_shoot_cov[:, i] + tau_feedback, np.zeros((2,)), q, qdot)
                X = ca.vertcat(q, qdot)
                dX = ca.vertcat(qdot, qdotdot)
                X_next = G_Euler_Explicit(X, dX, dt)
                q = X_next[:2]
                qdot = X_next[2:]

                # The feedforward and feedback torque squared are added to the cost function
                J = J + (tau_biological_shoot_cov[0, i] ** 2 + tau_biological_shoot_cov[1, i] ** 2) / (2 * nTOT + 1)
                J = J + (tau_feedback[0] ** 2 + tau_feedback[1] ** 2) / (2 * nTOT + 1)

            # Collect sigma points at the end of the interval
            if k == 0:
                sigma_next = X_next
            else:
                sigma_next = ca.horzcat(sigma_next, X_next)

        mean_next = ca.mtimes(sigma_next , WeightVec_M.T)
        opti.subject_to(mean_trajectory[:, j + 1] == mean_next)
        q = mean_trajectory[:2, j + 1]
        qdot = mean_trajectory[2:, j + 1]
        Y1k = sigma_next - mean_next
        P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
        L_next = L[:, j + 1]
        L_mat_next = f_LvectoLmat(L_next)
        opti.subject_to(1e3*(P_mat_next - ca.mtimes(L_mat_next,L_mat_next.T)) == 0) # why is this constraint giving so much issues?
        # L_mat = f_chol(P_mat_next + 0.00000001*np.eye(4))

    P_handPosition = f_dhandPosition_dq(mean_next[:2])*P_mat_next[:2,:2]*f_dhandPosition_dq(mean_next[:2]).T

    #
    opti.subject_to((P_handPosition[0, 0] - 0.01 ** 2)*1e3 < 0)
    opti.subject_to((P_handPosition[1, 1] - 0.01 ** 2)*1e3 < 0)
    q_final = mean_next[:2]
    qdot_final = mean_next[2:4]
    final_hand_position = f_handPosition(q_final)
    final_hand_velocity = f_handVelocity(q_final, qdot_final)
    opti.subject_to(final_hand_position > target - precision)
    opti.subject_to(final_hand_position < target + precision)

    opti.subject_to(final_hand_velocity > -0.0001)
    opti.subject_to(final_hand_velocity < 0.0001)
    opti.minimize(J)

    s_opts = {"linear_solver": "mumps",
              "hessian_approximation": "exact",
              "mu_strategy": "adaptive",
              "max_iter": 10000,
              "tol": 10 ** (-3),
              "constr_viol_tol": 10 ** (-6)}
    p_opts = {"expand": False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    tau_biological_sol = sol.value(tau_biological)
    k_feedback_sol = sol.value(k_feedback)
    mean_trajectory = sol.value(mean_trajectory)
    L_trajectory = sol.value(L)
    # L_final_sol = sol.value(L_final)
    L_mat_final = f_LvectoLmat(L_trajectory[:,-1])
    P_mat_final = L_mat_final*L_mat_final.T
    P_handPosition = f_dhandPosition_dq(mean_trajectory[:2,-1]) * P_mat_final[:2, :2] * f_dhandPosition_dq(mean_trajectory[:2,-1]).T

    plt.figure()
    plt.plot(mean_trajectory[0, :])
    plt.plot(mean_trajectory[1, :])
    plt.plot(mean_trajectory[2, :])
    plt.plot(mean_trajectory[3, :])
    q_final = mean_trajectory[:2,-1]
    qdot_final = mean_trajectory[2:,-1]
    final_hand_velocity = f_handVelocity(q_final, qdot_final)

    # plt.show()


    # Run multiple forward simulations with the solution
    number_of_forward_simulations = 10
    plt.figure()

    for i in range(number_of_forward_simulations):
        random_init = np.random.multivariate_normal(np.zeros((4,)), P_mat_init)
        q_forward = np.zeros((2, N+1))
        qdot_forward = np.zeros((2, N + 1))
        q_forward[:,0] = initial_q + random_init[:2]
        qdot_forward[:, 0] = random_init[2:]
        for j in range(N_cov):
            tau_biological_shoot_cov_j = tau_biological_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
            k_feedback_shoot_cov_j = k_feedback_sol[:, j * 4: (j + 1) * 4]
            mean_traj_j = mean_trajectory[:, j]
            q = q_forward[:, j*N_shoot_cov]
            qdot = qdot_forward[:, j*N_shoot_cov]
            X = ca.vertcat(q, qdot)
            tau_feedback = ca.mtimes(k_feedback_shoot_cov_j, X - mean_traj_j)
            for k in range(N_shoot_cov):
                qdotdot = f_armForwardDynamics(tau_biological_shoot_cov_j[:, k] + tau_feedback, np.zeros((2,)), q, qdot)
                X = ca.vertcat(q, qdot)
                dX = ca.vertcat(qdot, qdotdot)
                X_next = G_Euler_Explicit(X, dX, dt)
                q = X_next[:2]
                qdot = X_next[2:]
                q_forward[:, j*N_shoot_cov + k + 1] = np.reshape(q,(2,))
                qdot_forward[:, j*N_shoot_cov + k + 1] = np.reshape(qdot,(2,))
        plt.plot(q_forward[0, :])
        plt.plot(q_forward[1, :])

    plt.show()




    # Run forward simulation with solution
    if forward_sim_UT == True and run_shooting_OCP == True:

        finalTime = 1
        dt = 0.01
        N = int(finalTime / dt)
        N_cov = int(N / 10)
        N_shoot_cov = int(N / N_cov)

        precision = 0.01  # 1cm
        target = np.array((0, 0.52))  # in Cartesian coordinates
        initial_q = np.pi / 180 * np.array((30, 120))  # degrees to radian
        initial_qdot = np.pi / 180 * np.array((0, 0))  # degrees to radian
        endpoint_velocity = 0.001

        P_full = np.zeros((4,4,N_cov+1))

        q = initial_q
        qdot = initial_qdot
        mean_next = ca.vertcat(q, qdot)
        mean_previous = mean_next
        L_j = 0.01 * ca.vertcat(0.1, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0.1)
        L_mat = f_LvectoLmat(L_j)
        P_full[:, :, 0] = ca.mtimes(L_mat, L_mat.T)
        traj = np.ones((4,N_cov+1))
        traj[:,0] = np.reshape(mean_next,(4,))

        for j in range(N_cov):
            A = c * L_mat
            X = ca.vertcat(q, qdot)

            Y = ca.horzcat(X, X, X, X)
            sigma = ca.horzcat(X, Y + A, Y - A)
            tau_biological_shoot_cov =tau_biological_sol[:, j * N_shoot_cov: (j + 1) * N_shoot_cov]
            k_feedback_shoot_cov = k_feedback_sol[:, j * 4: (j + 1) * 4]

            for k in range(2 * nTOT + 1):
                q = sigma[:2, k]
                qdot = sigma[2:, k]
                X = ca.vertcat(q, qdot)
                tau_feedback = ca.mtimes(k_feedback_shoot_cov, X - mean_previous)

                for i in range(N_shoot_cov):
                    qdotdot = f_armForwardDynamics(tau_biological_shoot_cov[:, i] + tau_feedback, np.zeros((2,)), q, qdot)
                    X = ca.vertcat(q, qdot)
                    dX = ca.vertcat(qdot, qdotdot)
                    X_next = G_Euler_Explicit(X, dX, dt)
                    q = X_next[:2]
                    qdot = X_next[2:]

                if k == 0:
                    sigma_next = X_next
                    J = J + tau_biological_shoot_cov[0, i] ** 2 + tau_biological_shoot_cov[1, i] ** 2
                else:
                    sigma_next = ca.horzcat(sigma_next, X_next)
            mean_previous = mean_next
            mean_next = ca.mtimes(sigma_next, WeightVec_M.T)
            Y1k = sigma_next - mean_next
            P_mat_next = ca.mtimes(ca.mtimes(Y1k, ca.diag(WeightVec_C)), Y1k.T)
            P_full[:, :, j+1] = P_mat_next
            L_mat = f_chol(P_mat_next)
            traj[:, j + 1] = np.reshape(mean_next,(4,))

    P_handPosition = f_dhandPosition_dq(traj[:2, -1]) *  P_full[:2, :2, -1] * f_dhandPosition_dq(traj[:2, -1]).T
    final_hand_position = f_handPosition(traj[:2, -1])
    print(final_hand_position)
    print(handPos_det)
    print(P_handPosition[0,0] - 1e-6)
    plt.plot(traj[0, :])
    plt.plot(traj[1, :])
    plt.show()






