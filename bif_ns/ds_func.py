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
        ret[1] = pow(x[0], ds.p[4]) * (x[0]**2 - ds.p[1]**2)
    elif ds.var_param == 3:
        ret[0] = x[1]
        ret[1] = 0
    return ret


def dTdxdx(x, ds):
    ret = np.zeros((ds.xdim, ds.xdim, ds.xdim))
    ret[0][1][0] = ds.p[0] * ds.p[4] * (ds.p[4] - 1) * pow(x[0], ds.p[4] - 2) * (
        x[0]**2 - ds.p[1]**2) + 4 * x[0] * ds.p[0] * ds.p[4] * pow(x[0], ds.p[4] - 1) + 2 * ds.p[0] * pow(x[0], ds.p[4])
    return ret


def dTdxdlambda(x, ds):
    ret = np.zeros((ds.xdim, ds.xdim))
    if ds.var_param == 0:
        ret[1][0] = ds.p[4] * pow(x[0], ds.p[4] - 1) * \
            (x[0]**2 - ds.p[1]**2) + 2 * pow(x[0], ds.p[4] + 1)
    elif ds.var_param == 3:
        ret[0][1] = 1
    return ret


def store_state(v, ds):
    ds.xk[0] = v[0:ds.xdim]
    ds.p[ds.var_param] = v[ds.xdim]
    ds.theta = v[ds.xdim+1]

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
    ds.mu = complex(np.cos(ds.theta), np.sin(ds.theta))

    ds.chara_poly = ds.dTldx - ds.mu * np.eye(ds.xdim)


def dTldx(ds):
    return functools.partial(functools.reduce, np.matmul)(reversed(ds.dTdx))


def dTldlambda(ds):
    ds.dTkdlambda = [np.zeros(ds.xdim) for i in range(ds.period)]
    ret = ds.dTdlambda[0]
    ds.dTkdlambda[0] = ret
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

    for i in range(ds.xdim):
        for j in range(ds.period):
            ret[i] += ds.frwd_prod[j] @ ds.dTdxdx[j][i] @ ds.bkwd_prod[j] @ ds.bkwd_prod[j]
    # for i in range(ds.period):
    #     ret += ds.frwd_prod[i] @ ds.dTdxdx[i] @ ds.bkwd_prod[i] @ ds.bkwd_prod[i]

    return ret


def dTldxdlambda(ds):
    ret = np.zeros((ds.xdim, ds.xdim))
    temp = np.zeros((ds.xdim, ds.xdim))

    ret = ds.dTdxdlambda[0]
    for i in range(1, ds.period):
        for j in range(ds.xdim):
            temp[:, j] = ds.dTdxdx[i][j] @ ds.bkwd_prod[i] @ ds.dTkdlambda[i]
        # temp = ds.dTdxdx[i] @ ds.bkwd_prod[i] @ ds.dTkdlambda[i]
        ret = temp + ds.dTdx[i] @ ret + ds.dTdxdlambda[i] @ ds.bkwd_prod[i]

    return ret


def func_newton(ds):
    ret = np.zeros(ds.xdim + 2)
    chi = np.linalg.det(ds.chara_poly)

    ret[0:ds.xdim] = ds.xk[ds.period] - ds.xk[0]
    ret[ds.xdim] = chi.real
    ret[ds.xdim+1] = chi.imag

    return ret


def jac_newton(ds):
    ret = np.zeros((ds.xdim+2, ds.xdim+2))

    # dchidx = np.zeros(ds.xdim, dtype=np.complex)
    # for i in range(ds.xdim):
    #     dchidx[i] = det_derivative(ds.chara_poly, ds.dTldxdx[i], ds)
    dchidx = np.array([det_derivative(ds.chara_poly, ds.dTldxdx[i], ds)
                      for i in range(ds.xdim)])
    dchidlambda = det_derivative(ds.chara_poly, ds.dTldxdlambda, ds)
    dpolydtheta = complex(np.sin(ds.theta),
                          -np.cos(ds.theta)) * np.eye(ds.xdim)
    dchidtheta = det_derivative(ds.chara_poly, dpolydtheta, ds)

    ret[0:ds.xdim, 0:ds.xdim] = ds.dTldx - np.eye(ds.xdim)
    ret[0:ds.xdim, ds.xdim] = ds.dTldlambda
    ret[0:ds.xdim, ds.xdim+1] = np.zeros(ds.xdim)

    ret[ds.xdim, 0:ds.xdim] = dchidx.real
    ret[ds.xdim, ds.xdim] = dchidlambda.real
    ret[ds.xdim, ds.xdim+1] = dchidtheta.real

    ret[ds.xdim+1, 0:ds.xdim] = dchidx.imag
    ret[ds.xdim+1, ds.xdim] = dchidlambda.imag
    ret[ds.xdim+1, ds.xdim+1] = dchidtheta.imag

    return ret


def det_derivative(A, dA, ds):
    ret = 0+0j
    for i in range(0, ds.xdim):
        temp = A
        temp[:, i] = dA[:, i]
        ret += np.linalg.det(temp)
    return ret