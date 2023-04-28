import joblib.parallel
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from calibrate import *
from scipy.ndimage.filters import uniform_filter, generic_filter

import matcher.matcher_module


# def patch_scores(ps, ref, pad):
#     scores = []
#
#     for i in range(ps.shape[0]):
#         if i % 100_000 == 0:
#             print(i)
#
#         patch = ref[ps[i, 1]-pad:ps[i, 1]+pad+1, ps[i, 0]-pad:ps[i, 0]+pad+1, :].reshape((-1, 3))
#         scores.append(np.sum(np.std(patch, axis=0) / (np.mean(patch, axis=0) + 0.1)))
#
#     return np.array(scores)


def patch_scores_fast(ref, roi, pad):
    sub = ref[roi[1]-pad:roi[3]+pad+2, roi[0]-pad:roi[2]+pad+2, :].astype(np.float32)

    m = np.stack([uniform_filter(sub[:, :, i], size=2*pad+1) for i in range(3)], axis=2)
    m2 = np.stack([uniform_filter(sub[:, :, i]**2, size=2*pad+1) for i in range(3)], axis=2)
    s = np.sqrt(m2 - m**2)

    all_scores = np.sum(s / (m + 0.1), axis=2)
    scores = all_scores[pad:pad + roi[3] - roi[1] + 1, pad:pad + roi[2] - roi[0] + 1].ravel()

    return scores


def trace_ray_v(p, d, off):
    s = (off - p[2]) / d[:, 2]
    return p + d * s[:, None]


def map_to(p, off, ref_O, ref_B, O, B, mtx):
    p_u = cv2.undistortPoints(p.astype(np.float32).reshape((-1, 1, 2)), mtx, None).reshape((-1, 2))
    p_ray = np.concatenate([p_u, np.ones((p_u.shape[0], 1))], axis=1)
    w_ray = np.matmul(ref_B, p_ray.T).T

    w0 = trace_ray_v(ref_O, w_ray, 0)
    w1 = trace_ray_v(ref_O, w_ray, off)

    m0_ray = np.matmul(B.T, (w0 - O).T).T
    m1_ray = np.matmul(B.T, (w1 - O).T).T

    m0 = cv2.projectPoints(m0_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)
    m1 = cv2.projectPoints(m1_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)

    return m0, m1


def compute_diffs(r, w, img, ms, pad):
    r = r.astype(np.float32)
    diffs = []
    for mi in range(ms.shape[0]):
        patch = img[ms[mi, 1] - pad:ms[mi, 1] + pad + 1, ms[mi, 0] - pad:ms[mi, 0] + pad + 1, :]
        loss = w[:, :, None] * (r - patch) ** 2
        diffs.append(np.average(loss))

    return np.array(diffs)


def compute_diffs_fast(r, w, img, ms, pad):
    if ms.shape[0] == 0:
        print("Zero")
        return None

    diffs = np.zeros(ms.shape[0], dtype=np.float32)
    r = np.ascontiguousarray(r)

    matcher.matcher_module.compute_diffs(r, w, img, ms, pad, diffs)

    return diffs


def process_pair(ref, ps, filename, s0, s1, pad, max_h, plot=False):
    cache_filenane = filename[:-4] + ".npy"
    if os.path.exists(cache_filenane):
        print("Already exists:", cache_filenane)
        return np.load(cache_filenane)

    if ps.shape[0] == 0:
        return None

    # Define normalized 2D gaussian for weights matrix
    def gaus2d(x, y, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(
            -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))

    x1 = np.linspace(-pad, pad, 2*pad+1)
    x, y = np.meshgrid(x1, x1)
    w = gaus2d(x, y, sx=pad, sy=pad)
    w /= w[pad, pad]
    w = w.astype(np.float32)

    img = np.ascontiguousarray(cv2.imread(filename)[:, :, ::-1])

    if plot:
        plt.figure("Diffs " + filename, (16, 9))

    stats = []
    for pi in range(ps.shape[0]):
        if pi % 100_000 == 0:
            print(pi, "/", ps.shape[0], "-", filename)

        r = ref[ps[pi, 1]-pad:ps[pi, 1]+pad+1, ps[pi, 0]-pad:ps[pi, 0]+pad+1, :]
        n = np.max(np.abs(s0[pi, :] - s1[pi, :]))

        if n == 0:
            print("n == 0 in", filename)
            stats.append((1, 0, 0, 1000, 0))
            continue

        ms = np.stack([np.linspace(s0[pi, j], s1[pi, j], n+1, dtype=np.int32) for j in [0, 1]], axis=1)

        # diffs = compute_diffs(r, w, img, ms, pad)
        diffs = compute_diffs_fast(r, w, img, ms, pad)

        if diffs is None:
            stats.append((1, 0, 0, 1000, 0))
            continue

        mean, std = np.mean(diffs), np.std(diffs)
        x = np.linspace(0, max_h, diffs.shape[0], dtype=np.float32)

        if plot:
            plt.plot(x, diffs)

        span, am = np.max(diffs) - np.min(diffs), np.argmin(diffs)
        contrast, priority = span / (diffs[am] + 0.1), 1 / (diffs[am] + 0.1)

        stats.append((mean, std, x[am], diffs[am], contrast * priority))

    if len(stats) == 0:
        return None

    res = np.array(stats, dtype=np.float16)
    np.save(cache_filenane, res)

    return res


def roi_reconstruct(data_path, cam_calib, roi, max_h, ref_id=14, pad=25, score_thr=0.1,
                    save=None, plot=False, save_figures=None, **kw):
    corners = load_corners(data_path + "/undistorted/detected/corners.json")
    names = [name[:-4] + ".bmp" for name in sorted(corners.keys())]

    extrinsics = load_calibration(data_path + "/reconstructed/extrinsic.json")
    O, B = extrinsics["O"], extrinsics["B"]

    ref = np.ascontiguousarray(cv2.imread(data_path + "/undistorted/" + names[ref_id])[:, :, ::-1])
    h, w, _ = ref.shape

    assert pad < roi[0] and roi[2] < w - pad and \
           pad < roi[1] and roi[3] < h - pad and \
           roi[0] < roi[2] and roi[1] < roi[3], "Invalid ROI"

    px, py = np.meshgrid(np.linspace(roi[0], roi[2], roi[2] - roi[0] + 1, dtype=np.int32),
                         np.linspace(roi[1], roi[3], roi[3] - roi[1] + 1, dtype=np.int32))

    all_ps = np.stack([px, py], axis=2).reshape((-1, 2))
    print(px.shape, all_ps.shape)

    # scores = patch_scores(all_ps, ref, pad)
    # Or compute scores more quickly:
    scores = patch_scores_fast(ref, roi, pad)

    good_idx = np.nonzero(scores > score_thr)[0]
    good_ps = all_ps[good_idx, :]

    if plot:
        plt.figure("Mask", (16, 11))
        bad_ps = all_ps[np.nonzero(scores <= score_thr)[0], :]
        masked_ref = ref.copy()
        i, j = bad_ps[:, 1], bad_ps[:, 0]
        masked_ref[i, j, 0] = np.minimum(masked_ref[i, j, 0] + 100, 255)
        plt.imshow(masked_ref)

        plt.plot([pad, w-pad, w-pad, pad, pad], [pad, pad, h-pad, h-pad, pad], "g-")
        plt.plot([roi[0], roi[2], roi[2], roi[0], roi[0]],
                 [roi[1], roi[1], roi[3], roi[3], roi[1]], "b-")

        plt.title("ROI (blue) with Mask (red)")
        plt.tight_layout()

        if save_figures:
            plt.savefig("mask.png", dpi=400)

    # return

    jobs, all_idx = [], []
    for i, name in enumerate(names):
        if i == ref_id:
            continue

        s0, s1 = map_to(good_ps, max_h, O[ref_id], B[ref_id], O[i], B[i], cam_calib["new_mtx"])

        pad2 = pad + 2
        inside = (pad2 < s0[:, 0]) & (s0[:, 0] < w - pad2) & (pad2 < s0[:, 1]) & (s0[:, 1] < h - pad2) & \
                 (pad2 < s1[:, 0]) & (s1[:, 0] < w - pad2) & (pad2 < s1[:, 1]) & (s1[:, 1] < h - pad2)
        idx = np.nonzero(inside)[0]
        all_idx.append(idx)
        print(i, idx.shape)

        ps, s0, s1 = good_ps[idx, :], np.round(s0[idx, :]).astype(np.int32), np.round(s1[idx, :]).astype(np.int32)

        # if i == 0:
        # process_pair(ref, ps, data_path + "/undistorted/" + name, s0, s1, pad, max_h, plot=True)

        jobs.append(joblib.delayed(process_pair)(ref, ps, data_path + "/undistorted/" + name, s0, s1, pad, max_h))

    stats = joblib.Parallel(verbose=15, n_jobs=-2)(jobs)

    all_stats = np.zeros((good_ps.shape[0], len(names)-1, 1+5), dtype=np.float16)
    for i, stat in enumerate(stats):
        idx = all_idx[i]
        all_stats[:, i, 0] = -1

        if stat is not None:
            all_stats[idx, i, 1:] = stat
            all_stats[idx, i, 0] = 1

    height = np.zeros(good_ps.shape[0], dtype=np.float32)

    for i in range(good_ps.shape[0]):
        stats = all_stats[i, :, :]
        valid_stats = stats[stats[:, 0] > 0]
        good_stats = valid_stats[valid_stats[:, 2] > 0.1 * valid_stats[:, 1]]

        if good_stats.shape[0] > 0:
            xs, ms, ws = good_stats[:, 3], good_stats[:, 4], good_stats[:, 5]
            idx = ms < 2 * np.min(ms)
            xs, ws = xs[idx], ws[idx]

            if xs.shape[0] > 0:
                true_x = np.sum(xs * ws) / np.sum(ws)
                height[i] = true_x

    rw, rh = roi[2] - roi[0] + 1, roi[3] - roi[1] + 1

    all_height = np.zeros((rw * rh), dtype=np.float32)
    all_height[...] = None

    all_height[good_idx] = height
    all_height = all_height.reshape((rh, rw))

    if save:
        np.save("height.npy", all_height)
        all_height = np.load("height.npy")
        print(all_height)

    if plot:
        plt.figure("Height", (16, 8))
        plt.imshow(all_height)
        plt.title("Height in mm")
        plt.colorbar()
        plt.tight_layout()

        if save_figures:
            plt.savefig("height_map.png", dpi=240)

        h = all_height[~np.isnan(all_height)].ravel()
        plt.figure("Hist", (16, 9))
        plt.hist(h, bins=500)
        plt.title("Mean = %.2f (std = %.3f)" % (float(np.mean(h)), float(np.std(h))))
        plt.xlabel("Height, mm")
        plt.ylabel("Counts")
        plt.semilogy()
        plt.tight_layout()

        if save_figures:
            plt.savefig("height_hist.png", dpi=120)


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

    roi_reconstruct(data_path, cam_calib, (1550, 1450, 3550, 2450), 6, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (1700, 1700, 1750, 1750), 6, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (2990, 1990, 2996, 1994), 21, save=True, plot=True, save_figures=True)

    plt.show()
