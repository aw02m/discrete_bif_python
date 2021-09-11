#!/usr/bin/env python
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import pptools


def func(x, data):
    return np.array([
        data.dict['params'][3] * x[1],
        data.dict['params'][0] * x[0]**data.dict['params'][4] * (x[0]**2 - data.dict['params'][1]**2) + data.dict['params'][2] * x[0]
    ])


def main():

    data = pptools.init()
    x0 = data.dict['x0']

    running = True

    cnt = 0
    xlist = []
    ylist = []
    while True:
        if pptools.window_closed(data.ax) == True:
            sys.exit()
        x = func(x0, data)
        if np.linalg.norm(x, ord=2) > data.dict['explode']:
            x = x0
            explodeflag = True
        else:
            explodeflag = False
        xlist.append(x[0])
        ylist.append(x[1])
        x0 = x
        cnt += 1
        if (cnt > data.dict['break']):
            if explodeflag == True:
                print("exploded.")
            plt.plot(xlist, ylist, 'o', markersize=0.3,
                     color="black", alpha=data.dict['alpha'])
            x_hist = [list(e) for e in zip(xlist, ylist)]
            np.savetxt("out", np.array(x_hist), delimiter=' ')
            xlist.clear()
            ylist.clear()
            data.dict['x0'] = list(x0)
            cnt = 0
            plt.pause(0.01)


if __name__ == '__main__':
    main()
