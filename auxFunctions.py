import casadi as ca

def LvectoLmat(L):
    L_mat = ca.horzcat(L[:4, 0], ca.vertcat(0, L[4:7, 0]), ca.vertcat(0, 0, L[7:9, 0]), ca.vertcat(0, 0, 0, L[9, 0]))

    return L_mat


def LvectoLmat_10(L):
    L_mat = ca.horzcat(L[:10, 0], ca.vertcat(0, L[10:19, 0]), ca.vertcat(0, 0, L[19:27, 0]), ca.vertcat(0, 0, 0, L[27:34, 0]), ca.vertcat(0, 0, 0, 0, L[34:40, 0]), ca.vertcat(0, 0, 0, 0, 0, L[40:45, 0]), ca.vertcat(0, 0, 0, 0, 0, 0, L[45:49, 0]), ca.vertcat(0, 0, 0, 0, 0, 0, 0, L[49:52, 0]), ca.vertcat(0, 0, 0, 0, 0, 0, 0, 0, L[52:54, 0]), ca.vertcat(0, 0, 0, 0, 0, 0, 0, 0, 0, L[-1, 0]))

    return L_mat
