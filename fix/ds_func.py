import numpy as np
import functools


def T(x, ds):
    return np.array([
        ds.p[3] * x[1],
        ds.p[0] * x[0]**ds.p[4] * (x[0]**2 - ds.p[1]**2) + ds.p[2] * x[0]
    ])


def dTdx(x, ds):
    return np.array([
        [0, ds.p[3]],
        [ds.p[0] * ds.p[4] * x[0]**(ds.p[4]-1) *
         (x[0]**2 - ds.p[1]**2) +
         2 * ds.p[0] * x[0]**(ds.p[4]+1) + ds.p[2], 0]
    ])


def store_state(v, ds):
    ds.xk[0] = v[0:ds.xdim]

    for i in range(ds.period):
        ds.xk[i+1] = T(ds.xk[i], ds)
        ds.dTdx[i] = dTdx(ds.xk[i], ds)

    ds.dTldx = dTldx(ds)

    ds.eigvals = np.linalg.eigvals(ds.dTldx)


def dTldx(ds):
    return functools.partial(functools.reduce, np.matmul)(reversed(ds.dTdx))


def func_newton(ds):
    return ds.xk[ds.period] - ds.xk[0]


def jac_newton(ds):
    return ds.dTldx - np.eye(ds.xdim)
