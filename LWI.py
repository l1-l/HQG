
import os
import time

import numpy as np
from os.path import join
from collections import defaultdict
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

# 线性插值
def LinearInterpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # 按ID和帧排序
    output_ = input_.copy()
    '''线性插值'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))

    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # 同ID
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐框插值
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:  # 不同ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_


def LSmooth1(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        t = tracks[:, 0]
        x = tracks[:, 2]
        y = tracks[:, 3]
        w = tracks[:, 4]
        h = tracks[:, 5]

        xx = lowess(x, t, frac=0.07, it=0)[:, 1]
        yy = lowess(y, t, frac=0.07, it=0)[:, 1]
        ww = lowess(w, t, frac=0.07, it=0)[:, 1]
        hh = lowess(h, t, frac=0.07, it=0)[:, 1]

        output_.extend([
            [t[i], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1 , -1] for i in range(len(t))
        ])
    return output_


def LWInterpolation(path_in, path_out, interval, tau):
    input_ = np.loadtxt(path_in, delimiter=',')
    li = LinearInterpolation(input_, interval)
    lwi = LSmooth1(li, tau)
    np.savetxt(path_out, lwi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')


if __name__ == '__main__':
    path = "/home/dell/SparseTrack20/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results/"
    for txt in os.listdir(path):
        LWInterpolation(path + txt, path + txt, 30, 12)
        print(txt + '============= ok ===============')
