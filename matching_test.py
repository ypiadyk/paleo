import json
import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from utils import *
from calibrate import *


def trace_ray(T, R, p, d):
    A = np.stack((R[:, 0], R[:, 1], -d), axis=1)
    b = p - T
    uvt = np.matmul(np.linalg.inv(A), b)
    return p + uvt[2]*d


def trace_ray_v(p, d, off):
    s = (off - p[2]) / d[:, 2]
    return p + d * s[:, None]


def map_to(p, off, ref_O, ref_B, O, B, mtx, plot=False):
    p_u = cv2.undistortPoints(p.astype(np.float32).reshape((-1, 1, 2)), mtx, None).reshape((-1, 2))
    p_ray = np.concatenate([p_u, np.ones((p_u.shape[0], 1))], axis=1)
    w_ray = np.matmul(ref_B, p_ray.T).T

    # T, R = np.zeros(3), np.eye(3)
    # w0 = trace_ray(T, R, ref_O, w_ray[0, :]).reshape((-1, 3))
    # w1 = trace_ray(T + np.array([0, 0, off]), R, ref_O, w_ray[0, :]).reshape((-1, 3))

    w0 = trace_ray_v(ref_O, w_ray, 0)
    w1 = trace_ray_v(ref_O, w_ray, off)

    m0_ray = np.matmul(B.T, (w0 - O).T).T
    m1_ray = np.matmul(B.T, (w1 - O).T).T

    m0 = cv2.projectPoints(m0_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)
    m1 = cv2.projectPoints(m1_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)

    return m0, m1


def match_single(ref, w, img, ps, pad):
    diffs = []
    for i in range(ps.shape[0]):
        patch = img[ps[i, 1]-pad:ps[i, 1]+pad+1, ps[i, 0]-pad:ps[i, 0]+pad+1, :]
        # loss = np.abs(ref - patch)
        # loss = w[:, :, None] * np.abs(ref - patch)
        # loss = (ref - patch)**2
        loss = w[:, :, None] * (ref - patch)**2
        # loss = (w[:, :, None] * (ref - patch))**2
        # loss = ref * patch.astype(np.float32)
        diffs.append(np.average(loss))

    return np.array(diffs)


def match_img_pair(ref, img, p, s0, s1, pad):
    matches = []

    # define normalized 2D gaussian
    def gaus2d(x, y, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(
            -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

    x1 = np.linspace(-pad, pad, 2*pad+1)
    x, y = np.meshgrid(x1, x1)
    w = gaus2d(x, y, sx=pad, sy=pad)
    w /= w[pad, pad]

    tot_n = 0
    for i in range(p.shape[0]):
        r = ref[p[i, 1]-pad:p[i, 1]+pad+1, p[i, 0]-pad:p[i, 0]+pad+1, :].astype(np.float32)
        n = np.max(np.abs(s0[i, :] - s1[i, :]))
        tot_n += n
        ps = np.stack([np.linspace(s0[i, j], s1[i, j], n+1, dtype=np.int32) for j in [0, 1]], axis=1)
        matches.append(match_single(r, w, img, ps, pad))

    print(tot_n)
    return matches


def do_matching(data_path, cam_calib, save=None, plot=False, save_figures=None, **kw):
    corners = load_corners(data_path + "/undistorted/detected/corners.json")
    names = sorted(corners.keys())
    names = [name[:-4] + ".bmp" for name in names]

    extrinsics = load_calibration(data_path + "/reconstructed/extrinsic.json")
    O, B = extrinsics["O"], extrinsics["B"]

    # imgs = [cv2.imread(data_path + "/undistorted/" + name)[:, :, ::-1] for name in names]
    imgs = [None for name in names]

    ref, max_h, pad = 14, 21, 25
    # px, py = np.meshgrid(np.linspace(2000, 3500, 7), np.linspace(1750, 2750, 5))#, indexing="ij")
    px, py = np.meshgrid(np.linspace(2990, 2996, 7), np.linspace(1990, 1994, 5))#, indexing="ij")
    all_p = np.stack([px, py], axis=2).reshape((-1, 2)).astype(np.int32)

    imgs[ref] = cv2.imread(data_path + "/undistorted/" + names[ref])[:, :, ::-1]

    print(len(imgs), imgs[ref].shape)
    plt.figure("Ref", (16, 9))
    plt.imshow(imgs[ref])
    plt.plot(all_p[:, 0], all_p[:, 1], "r+")

    for i in range(all_p.shape[0]):
        x, y = all_p[i, 0], all_p[i, 1]
        plt.plot([x-pad, x+pad, x+pad, x-pad, x-pad], [y-pad, y-pad, y+pad, y+pad, y-pad], "r-")

    plt.tight_layout()

    all_matches = [{} for i in range(all_p.shape[0])]
    for i in range(len(imgs)):
        if i == ref:
            continue

        imgs[i] = cv2.imread(data_path + "/undistorted/" + names[i])[:, :, ::-1]

        s0, s1 = map_to(all_p, max_h, O[ref], B[ref], O[i], B[i], cam_calib["new_mtx"], plot=(plot and (i == 1)))

        w, h = imgs[ref].shape[1], imgs[ref].shape[0]
        inside = (pad < s0[:, 0]) & (s0[:, 0] < w - pad) & (pad < s0[:, 1]) & (s0[:, 1] < h - pad) & \
                 (pad < s1[:, 0]) & (s1[:, 0] < w - pad) & (pad < s1[:, 1]) & (s1[:, 1] < h - pad)
        idx = np.nonzero(inside)[0]
        print(i, idx.shape)

        good_p, s0, s1 = all_p[idx, :], np.round(s0[idx, :]).astype(np.int32), np.round(s1[idx, :]).astype(np.int32)

        matches = match_img_pair(imgs[ref], imgs[i], good_p, s0, s1, pad)
        # print(len(matches))

        for j, m in zip(idx, matches):
            all_matches[j][i] = m

        if i in [ref-2, ref+2]:
            plt.figure(str(i), (16, 9))
            plt.imshow(imgs[i])
            plt.plot(s0[:, 0], s0[:, 1], "rx")
            plt.plot(s1[:, 0], s1[:, 1], "gx")

            for j in range(s0.shape[0]):
                x, y = s0[j, 0], s0[j, 1]
                plt.plot([x - pad, x + pad, x + pad, x - pad, x - pad], [y - pad, y - pad, y + pad, y + pad, y - pad], "r-")

            plt.tight_layout()

        # if i == 2:
        #     break

    # print(len(all_matches), all_matches)

    if plot:
        plt.figure("Diffs", (21, 13))
        for i in range(7*5):
            # print(all_p[i, :])
            plt.subplot(5, 7, i+1, title="%d, %d" % (i // 7, i % 7))

            xs, ms, ws = [], [], []
            y_min, y_max = 1e+9, 0

            for mi, match in all_matches[i].items():
                mean, std = np.mean(match), np.std(match)
                if std < 0.1 * mean:
                    continue

                x = np.linspace(0, max_h, match.shape[0])

                span = np.max(match) - np.min(match)
                am = np.argmin(match)

                contrast = span / (match[am] + 0.1)
                priority = 1 / (match[am] + 0.1)

                xs.append(x[am])
                ms.append(match[am])
                ws.append(contrast * priority)

                plt.plot(x, match, label=str(mi))
                y_min, y_max = min(y_min, np.min(match)), max(y_max, np.max(match))

            if len(ms) > 0:
                xs, ms, ws = np.array(xs), np.array(ms), np.array(ws)
                idx = ms < 2 * np.min(ms)
                xs, ws = xs[idx], ws[idx]

                if xs.shape[0] > 0:
                    true_x = np.sum(xs * ws) / np.sum(ws)
                    print(i // 7, i % 7, true_x)

                    plt.plot([true_x, true_x], [y_min * 0.9, y_max * 1.1], "k--")

            plt.xlim([-1, max_h])
            # plt.ylim([0, 200])
            # if i == 0:
            #     plt.legend()

        plt.tight_layout()


if __name__ == "__main__":
    calib_data = "D:/paleo-data/CALIBRATION BOARD 1e2/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 3e4/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 5/"
    # calib_data = "D:/paleo-data/test-calib/"

    cam_calib = load_calibration(calib_data + "/calibrated/geometry.json")

    # data_path = "D:/paleo-data/1 - FLAT OBJECT 1/"
    data_path = "D:/paleo-data/2 - FLAT OBJECT 2/"
    # data_path = "D:/paleo-data/3 - IRREGULAR OBJECT 1/"
    # data_path = "D:/paleo-data/4 - IRREGULAR OBJECT 2/"
    # data_path = "D:/paleo-data/5 - BOX/"
    # data_path = "D:/paleo-data/test-scan-1/"
    # data_path = "D:/paleo-data/test-scan-2/"

    do_matching(data_path, cam_calib, save=True, plot=True, save_figures=True)

    plt.show()