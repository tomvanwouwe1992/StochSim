def G_Trapezoidal_Implicit(X, X_next, dX, dX_next, dt):

    G_error = X_next - (X + dt * (dX + dX_next) / 2)

    return G_error


def G_Euler_Explicit(X, dX, dt):

    X_next = X + dt * dX

    return X_next

