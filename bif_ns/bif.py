import sys
import time
import numpy as np
import json
import dynamical_system
import ds_func


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    ds = dynamical_system.DynamicalSystem(json_data)

    vp = np.append(ds.x0, [ds.p[ds.var_param], ds.theta])

    for p in range(ds.inc_iter):
        start = time.time()
        for i in range(ds.max_iter):
            ds_func.store_state(vp, ds)
            F = ds_func.func_newton(ds)
            J = ds_func.jac_newton(ds)
            vn = np.linalg.solve(J, -F) + vp
            norm = np.linalg.norm(vn - vp)
            if (norm < ds.eps):
                end = time.time()
                dur = end - start
                msec = dur
                print("**************************************************")
                print(str(p) + " : converged (iter = " +
                      str(i+1) + ", time = " + str(msec)[0:8] + "[sec])")
                print("params : " + str(ds.p))
                print("x0     : " + str(vn[0:ds.xdim]))
                print("(Re(μ), Im(μ)), abs(μ) :")
                for k in range(ds.xdim):
                    print(str(ds.eigvals[k]) + ", " +
                          str(abs(ds.eigvals[k]))[0:6])
                print("**************************************************")
                vp = vn
                ds.p[ds.inc_param] += ds.delta_inc
                break
            elif (norm > ds.explode):
                print("explode")
                print(F)
                sys.exit()
            vp = vn
        else:
            print("iter over")
            print(F)
            exit()


if __name__ == '__main__':
    main()
