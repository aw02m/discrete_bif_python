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


def dTdlambda(x, ds):
    ret = np.zeros(ds.xdim)
    if ds.var_param == 0:
        ret[0] = 0
        ret[1] = pow(x[0], ds.p[4]) * (x[0]*x[0] - ds.p[1]*ds.p[1])
    elif ds.var_param == 3:
        ret[0] = x[1]
        ret[1] = 0
    return ret


def dTdxdx(x, ds):
    ret = np.zeros((ds.xdim, ds.xdim, ds.xdim))
    ret[0][1][0] = ds.p[0] * ds.p[4] * (ds.p[4] - 1) * pow(x[0], ds.p[4] - 2) * (
        x[0]*x[0] - ds.p[1]*ds.p[1]) + 4 * x[0] * ds.p[0] * ds.p[4] * pow(x[0], ds.p[4] - 1) + 2 * ds.p[0] * pow(x[0], ds.p[4])
    return ret


def dTdxdlambda(x, ds):
    ret = np.zeros((ds.xdim, ds.xdim))
    if ds.var_param == 0:
        ret[1][0] = ds.p[4] * pow(x[0], ds.p[4] - 1) * (x[0]
                                                        * x[0] - ds.p[1]*ds.p[1]) + 2 * pow(x[0], ds.p[4] + 1)
    elif ds.var_param == 3:
        ret[0][1] = 1
    return ret


def store_state(v, ds):
    ds.xk[0] = v[0:ds.xdim]

    for i in range(ds.period):
        ds.xk[i+1] = T(ds.xk[i], ds)
        ds.dTdx[i] = dTdx(ds.xk[i], ds)
        ds.dTdlambda[i] = dTdlambda(ds.xk[i], ds)
        ds.dTdxdx[i] = dTdxdx(ds.xk[i], ds)
        ds.dTdxdlambda[i] = dTdxdlambda(ds.xk[i], ds)

    ds.dTldx = dTldx(ds)
    ds.dTldlambda = dTldlambda(ds)
    ds.dTldxdx = dTldxdx(ds)
    ds.dTldxdlambda = dTldxdlambda(ds)

    ds.eigvals = np.linalg.eigvals(ds.dTldx)

    ds.chara_poly = ds.dTldx + np.eye(ds.xdim)


def dTldx(ds):
    return functools.partial(functools.reduce, np.matmul)(reversed(ds.dTdx))


def dTldlambda(ds):
    ret = ds.dTdlambda[0]
    for i in range(1, ds.period):
        ret = ds.dTdx[i] @ ret + ds.dTdlambda[i]
        ds.dTkdlambda[i] = ret
    return ret


def dTldxdx(ds):
    ret = np.zeros((ds.xdim, ds.xdim, ds.xdim))
    ds.frwd_prod = [np.eye(ds.xdim)
                    for i in range(ds.period)]
    ds.bkwd_prod = [np.eye(ds.xdim)
                    for i in range(ds.period)]
    for i in range(ds.period):
        for j in np.arange(ds.period-1, -1, -1):
            if(j > i):
                ds.frwd_prod[i] = ds.frwd_prod[i] @ ds.dTdx[j]
            elif(j < i):
                ds.bkwd_prod[i] = ds.bkwd_prod[i] @ ds.dTdx[j]
    for i in range(ds.period):
        ret += ds.frwd_prod[i] @ ds.dTdxdx[i] @ ds.bkwd_prod[i] @ ds.bkwd_prod[i]
    return ret


def dTldxdlambda(ds):
    ret = np.zeros((ds.xdim, ds.xdim))
    temp = np.zeros((ds.xdim, ds.xdim))
    ret = ds.dTdxdlambda[0]
    for i in range(1, ds.period):
        for j in range(ds.xdim):
            temp[:, j] = ds.dTdxdx[i][j] @ ds.bkwd_prod[i] @ ds.dTkdlambda[i]
        ret = temp + ds.dTdx[i] * ret + ds.dTdxdlambda[i] @ ds.bkwd_prod[i]
    return ret


def func_newton(ds):
    ret = np.zeros(ds.xdim + 1)
    ret[0:ds.xdim] = ds.xk[ds.period] - ds.xk[0]
    ret[ds.xdim] = np.linalg.det(ds.chara_poly)
    return ret


def jac_newton(ds):
    ret = np.zeros((ds.xdim+1, ds.xdim+1))
    ret[0:ds.xdim, 0:ds.xdim] = ds.dTldx - np.eye(ds.xdim)
    ret[0:ds.xdim, ds.xdim] = ds.dTldlambda
    for i in range(ds.xdim):
        ret[ds.xdim, i] = det_derivative(ds.chara_poly, ds.dTldxdx[i], ds)
    ret[ds.xdim, ds.xdim] = det_derivative(ds.chara_poly, ds.dTldxdlambda, ds)
    return ret


def det_derivative(A, dA, ds):
    temp = np.zeros((ds.xdim, ds.xdim))
    ret = 0
    for i in range(ds.xdim):
        temp = A.copy()
        temp[:, i] = dA[:, i]
        ret += np.linalg.det(temp)
    return ret