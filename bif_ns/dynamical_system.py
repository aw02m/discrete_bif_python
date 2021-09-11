import numpy as np
import cmath

class DynamicalSystem:
    x0 = []
    xdim = 0
    period = 0

    inc_param = 0
    var_param = 0
    delta_inc = 0.0
    inc_iter = 0
    max_iter = 0
    dif_strip = 0.0
    eps = 0.0
    explode = 0.0

    xk = []
    params = []
    dTdx = []
    dTdlambda = []
    dTdxdx = []
    dtdxdlambda = []
    dTldx = []
    dTldlambda = []
    dTldxdx = []
    dTldxdlambda = []

    dTkdlambda = []
    frwd_prod = []
    bkwd_prod = []

    chara_poly = []

    sigma = 0
    omega = 0
    mu = 0
    theta = 0

    eigvals = []

    def __init__(self, json):
        self.x0 = np.array(json['x0'])
        self.p = np.array(json['params'])
        self.period = json['period']
        self.xdim = len(self.x0)
        self.inc_param = json['inc_param']
        self.var_param = json['var_param']
        self.delta_inc = json['delta_inc']
        self.inc_iter = json['inc_iter']
        self.max_iter = json['max_iter']
        self.dif_strip = json['dif_strip']
        self.eps = json['eps']
        self.explode = json['explode']
        self.sigma = json['sigma']
        self.omega = json['omega']
        self.mu = complex(self.sigma, self.omega)
        self.theta = cmath.phase(self.mu)
        self.xk = [np.zeros(self.xdim) for i in range(self.period + 1)]
        self.xk[0] = self.x0
        self.dTdx = [np.zeros((self.xdim, self.xdim))
                     for i in range(self.period)]
        self.dTdlambda = [np.zeros(self.xdim) for i in range(self.period)]
        self.dTdxdx = [np.zeros((self.xdim, self.xdim, self.xdim))
                       for i in range(self.period)]
        self.dTdxdlambda = [np.zeros((self.xdim, self.xdim))
                            for i in range(self.period)]
        self.dTkdlambda = [np.zeros(self.xdim) for i in range(self.period)]
        self.frwd_prod = [np.eye(self.xdim)
                          for i in range(self.period)]
        self.bkwd_prod = [np.eye(self.xdim)
                          for i in range(self.period)]
