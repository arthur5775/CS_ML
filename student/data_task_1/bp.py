import numpy as np
from scipy.optimize import linprog


def bp_solver(A, y):
    """ Solve the Basis pursuit problem
    """
    
    x_dim, y_dim = A.shape[1], y.shape[0]
    eye = np.eye(x_dim)

    obj = np.concatenate([np.zeros(x_dim), np.ones(x_dim)])

    lhs_ineq = np.concatenate([np.concatenate([eye, -eye], axis=1), np.concatenate([-eye, -eye], axis=1)], axis=0)
    rhs_ineq = np.zeros(2 * x_dim)

    lhs_eq = np.concatenate([A, np.zeros((y_dim, x_dim))], axis=1)
    rhs_eq = y

    bnd = [*((None, None) for _ in range(x_dim)), *((0, None) for _ in range(x_dim))]

    res = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="highs")
    return res.x[:x_dim]
