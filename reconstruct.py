import matplotlib.pyplot as plt

from utils import *
from calibrate import *
from scipy.ndimage.filters import uniform_filter

import matcher.matcher_module as mm
simd_pad = 5  # Results in 15 extra pixels


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


def map_to(p, off_0, off_1, ref_O, ref_B, O, B, mtx):
    p_u = cv2.undistortPoints(p.astype(np.float32).reshape((-1, 1, 2)), mtx, None).reshape((-1, 2))
    p_ray = np.concatenate([p_u, np.ones((p_u.shape[0], 1))], axis=1)
    w_ray = np.matmul(ref_B, p_ray.T).T

    w0 = trace_ray_v(ref_O, w_ray, off_0)
    w1 = trace_ray_v(ref_O, w_ray, off_1)

    m0_ray = np.matmul(B.T, (w0 - O).T).T
    m1_ray = np.matmul(B.T, (w1 - O).T).T

    m0 = cv2.projectPoints(m0_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)
    m1 = cv2.projectPoints(m1_ray, np.zeros(3), np.zeros(3), mtx, None)[0].reshape(-1, 2)

    return m0, m1


def compute_diffs(r, w, img, ms, pad):
    diffs = []

    for mi in range(ms.shape[0]):
        patch = img[ms[mi, 1] - pad:ms[mi, 1] + pad + 1, ms[mi, 0] - pad:ms[mi, 0] + pad + 1, :]
        diffs.append(np.average(w[:, :, None] * (r - patch) ** 2))

    return np.array(diffs)


def gaus2d(x, y, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * \
           np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def process_pair(ref, ps, filename, s0, s1, pad, min_h, max_h, steps, plot=False):
    prefix = "/%d_%d_%d/" % (min_h, max_h, steps)
    ensure_exists(os.path.dirname(filename) + prefix)
    cache_filenane = os.path.dirname(filename) + prefix + os.path.basename(filename)[:-4] + ".npy"
    if os.path.exists(cache_filenane):
        print("Already exists:", cache_filenane)
        return np.load(cache_filenane)

    if ps.shape[0] == 0:
        return None

    # Arrays must be contiguous for use with C extensions
    img = np.ascontiguousarray(cv2.imread(filename)[:, :, ::-1])

    # Prep weights matrix
    x1 = np.linspace(-pad, pad, 2*pad+1)
    x, y = np.meshgrid(x1, x1)
    w = gaus2d(x, y, sx=pad, sy=pad)
    w /= w[pad, pad]
    w = w.astype(np.float32)
    # print("w:", w.shape, w.dtype, "\n", w)

    # Prepare 16 bit packed and padded version for simd optimized functions
    w_f = np.concatenate([w, np.zeros((w.shape[0], simd_pad), dtype=np.float32)], axis=1)
    w_f = np.repeat(w_f[:, :, None], 3, axis=2)
    w_f16 = np.ascontiguousarray(w_f.astype(np.float16)).view(np.uint16)
    # print("w_f16:", w_f16.shape, w_f16.dtype)

    if plot:
        plt.figure("Diffs " + filename, (16, 9))

    stats = []
    for pi in range(ps.shape[0]):
        if pi % 100_000 == 0:
            print(pi, "/", ps.shape[0], "-", filename)

        n = np.max(np.abs(s0[pi, :] - s1[pi, :]))
        n = min(n, steps * (max_h - min_h))

        if n == 0:
            print("n == 0 in", filename)
            stats.append((1, 0, 0, 1000, 0))
            continue

        ms = np.stack([np.linspace(s0[pi, j], s1[pi, j], n+1, dtype=np.int32) for j in [0, 1]], axis=1)

        # r = ref[ps[pi, 1] - pad:ps[pi, 1] + pad + 1, ps[pi, 0] - pad:ps[pi, 0] + pad + 1, :].astype(np.float32)
        # diffs = compute_diffs(r, w, img, ms, pad)  # Slow Python/numpy speed

        diffs = np.zeros(ms.shape[0], dtype=np.float64)
        # mm.compute_diffs(ref, w, ps[pi, :], img, ms, pad, diffs)  # 10x speedup (C)
        mm.compute_diffs_avx2(ref, w_f16, ps[pi, :], img, ms, pad, diffs)  # 25x speedup (C + AVX2 SIMD)
        # mm.compute_diffs_avx512(ref, w_f16, ps[pi, :], img, ms, pad, diffs)  # 30x speedup (C + AVX512 SIMD)

        mean, std = np.mean(diffs), np.std(diffs)
        x = np.linspace(min_h, max_h, diffs.shape[0], dtype=np.float32)

        if plot:
            plt.plot(x, diffs)

        span, am = np.max(diffs) - np.min(diffs), np.argmin(diffs)
        contrast = (span + 0.1) / (diffs[am] + 0.1)
        priority = 1 / (diffs[am] + 0.1)
        resolution = float(n)

        stats.append((mean, std, x[am], diffs[am], contrast * priority * resolution))

    if len(stats) == 0:
        return None

    res = np.array(stats, dtype=np.float16)
    np.save(cache_filenane, res)

    return res


def roi_reconstruct(data_path, cam_calib, roi, min_h, max_h, steps=15, ref_id=None, pad=10, score_thr=0.1,
                    save=None, plot=False, save_figures=None, **kw):
    corners = load_corners(data_path + "/undistorted/detected/corners.json")
    names = [name[:-4] + ".bmp" for name in sorted(corners.keys())]
    ref_id = ref_id or (len(names) // 2 - 1)

    extrinsics = load_calibration(data_path + "/reconstructed/extrinsic.json")
    O, B = extrinsics["O"], extrinsics["B"]

    ref = np.ascontiguousarray(cv2.imread(data_path + "/undistorted/" + names[ref_id])[:, :, ::-1])
    h, w, _ = ref.shape

    # Extra simd_pad horizontal pixels (RGB) for SIMD padding
    assert pad < roi[0] and roi[2] < w - pad - 1 - simd_pad and \
           pad < roi[1] and roi[3] < h - pad - 1 and \
           roi[0] < roi[2] and roi[1] < roi[3], "Invalid ROI"

    px, py = np.meshgrid(np.linspace(roi[0], roi[2], roi[2] - roi[0] + 1, dtype=np.int32),
                         np.linspace(roi[1], roi[3], roi[3] - roi[1] + 1, dtype=np.int32))

    all_ps = np.stack([px, py], axis=2).reshape((-1, 2))
    print(px.shape, all_ps.shape)

    scores = patch_scores_fast(ref, roi, pad)
    # c, r = np.meshgrid(np.arange(roi[0], roi[2]+1), np.arange(roi[1], roi[3]+1))
    # all_coords = np.stack([r, c], axis=2).reshape((-1, 2))
    # print(all_ps - all_coords)

    good_idx = np.nonzero(scores > score_thr)[0]
    good_ps = all_ps[good_idx, :]

    good_rays = np.matmul(B[ref_id], img_to_ray(good_ps, cam_calib["new_mtx"]).T).T
    good_0s = trace_ray_v(O[ref_id], good_rays, 0)
    good_dirs = O[ref_id] - good_0s
    good_dirs /= good_dirs[:, 2][:, None]
    # good_coords = all_coords[good_idx, :]

    if plot:
        plt.figure("Scores", (16, 11))
        plt.imshow(scores.reshape((-1, roi[2]-roi[0]+1)))
        plt.colorbar()
        plt.tight_layout()

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
            plt.savefig(data_path + "/reconstructed/mask.jpg", dpi=300)

    # return

    map_jobs = [joblib.delayed(map_to)(good_ps, min_h, max_h, O[ref_id], B[ref_id], O[i], B[i], cam_calib["new_mtx"])
                for i in range(len(names)) if i != ref_id]

    ss = joblib.Parallel(verbose=15, n_jobs=-1)(map_jobs)

    print("Mapped points")

    jobs, all_idx, lengths = [], [], []
    for i, name in enumerate(names):
        if i == ref_id:
            continue

        s0, s1 = ss[i if i < ref_id else i-1]

        pad_x, pad_y = pad + 1 + simd_pad, pad + 1
        inside = (pad < s0[:, 0]) & (s0[:, 0] < w - pad_x) & (pad < s0[:, 1]) & (s0[:, 1] < h - pad_y) & \
                 (pad < s1[:, 0]) & (s1[:, 0] < w - pad_x) & (pad < s1[:, 1]) & (s1[:, 1] < h - pad_y)
        idx = np.nonzero(inside)[0]
        all_idx.append(idx)
        print(i, idx.shape)

        ps, s0, s1 = good_ps[idx, :], np.round(s0[idx, :]).astype(np.int32), np.round(s1[idx, :]).astype(np.int32)

        jobs.append(joblib.delayed(process_pair)(ref, ps, data_path + "/undistorted/" + name, s0, s1, pad, min_h, max_h, steps))
        lengths.append(idx.shape[0])

    order = np.argsort(np.array(lengths)).tolist()
    jobs = [jobs[i] for i in reversed(order)]
    all_idx = [all_idx[i] for i in reversed(order)]

    stats = joblib.Parallel(verbose=15, n_jobs=-1)(jobs)

    print("Computed stats")

    all_stats = np.zeros((good_ps.shape[0], len(names)-1, 1+5), dtype=np.float32)
    for i, stat in enumerate(stats):
        idx = all_idx[i]
        all_stats[:, i, 0] = -1

        if stat is not None:
            all_stats[idx, i, 1:] = stat
            all_stats[idx, i, 0] = 1

    print("Merged stats")

    height = np.zeros(good_ps.shape[0], dtype=np.float32)
    p3d = np.zeros((good_ps.shape[0], 3), dtype=np.float32)
    colors = np.zeros((good_ps.shape[0], 3), dtype=np.float32)
    counts = np.zeros_like(height, dtype=np.int32)

    for i in range(good_ps.shape[0]):
        stats = all_stats[i, :, :]
        valid_stats = stats[stats[:, 0] > 0]
        good_stats = valid_stats[valid_stats[:, 2] > 0.2 * valid_stats[:, 1]]

        if good_stats.shape[0] > 0:
            xs, ms, ws = good_stats[:, 3], good_stats[:, 4], good_stats[:, 5]
            idx = ms < 4 * np.min(ms)
            xs, ws = xs[idx], ws[idx]

            if xs.shape[0] > 0:
                true_x = np.sum(xs * ws) / np.sum(ws)
                c, r = good_ps[i, :]
                height[i] = true_x
                counts[i] = xs.shape[0]
                p3d[i, :] = good_0s[i, :] + good_dirs[i, :] * true_x
                colors[i, :] = ref[r, c, :] / 255.0

    rw, rh = roi[2] - roi[0] + 1, roi[3] - roi[1] + 1

    all_height = np.zeros((rw * rh), dtype=np.float32)
    all_counts = np.zeros((rw * rh), dtype=np.float32)
    all_height[...] = None
    all_counts[...] = None

    all_height[good_idx] = height
    all_counts[good_idx] = counts
    all_height = all_height.reshape((rh, rw))
    all_counts = all_counts.reshape((rh, rw))

    print("Computed heights")

    if save:
        save_ply(data_path + "/reconstructed/points.ply", p3d, colors=colors)

        np.save(data_path + "/reconstructed/height.npy", all_height)
        all_height = np.load(data_path + "/reconstructed/height.npy")
        print(all_height)

    # all_height = np.load("height.npy")

    if plot:
        plt.figure("Height", (16, 8))
        plt.imshow(all_height)
        plt.title("Height in mm")
        plt.colorbar()
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/reconstructed/height_map.png", dpi=240)

        plt.figure("Counts", (16, 8))
        plt.imshow(all_counts)
        plt.title("Image counts")
        plt.colorbar()
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/reconstructed/height_counts.png", dpi=240)

        h = all_height[~np.isnan(all_height)].ravel()
        h = np.minimum(h, 30)
        plt.figure("Hist", (16, 9))
        plt.hist(h, bins=500)
        plt.title("Mean = %.2f (std = %.3f)" % (float(np.mean(h)), float(np.std(h))))
        plt.xlabel("Height, mm")
        plt.ylabel("Counts")
        plt.semilogy()
        plt.tight_layout()

        if save_figures:
            plt.savefig(data_path + "/reconstructed/height_hist.png", dpi=120)

    print("Done reconstructing")


if __name__ == "__main__":
    calib_data = "D:/paleo-data/CALIBRATION BOARD 1e2/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 3e4/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 5/"
    # calib_data = "D:/paleo-data/test-calib/"
    # calib_data = "D:/paleo_scans/calib/"
    calib_data = "D:/paleo_scans/calib_17/"

    cam_calib = load_calibration(calib_data + "/calibrated/geometry.json")

    # data_path = "D:/paleo-data/1 - FLAT OBJECT 1/"
    # data_path = "D:/paleo-data/2 - FLAT OBJECT 2/"
    # data_path = "D:/paleo-data/3 - IRREGULAR OBJECT 1/"
    # data_path = "D:/paleo-data/4 - IRREGULAR OBJECT 2/"
    # data_path = "D:/paleo-data/5 - BOX/"
    # data_path = "D:/paleo-data/test-scan-1/"
    # data_path = "D:/paleo-data/test-scan-2/"
    # data_path = "D:/paleo_scans/scan_0/"
    # data_path = "D:/paleo_scans/scan_1/"
    # data_path = "D:/paleo_scans/scan_2/"
    data_path = "D:/paleo_scans/scan_17/"

    # roi_reconstruct(data_path, cam_calib, (1550, 1450, 3550, 2450), 6, save=True, plot=True, save_figures=False)
    # roi_reconstruct(data_path, cam_calib, (1700, 1700, 1750, 1750), 6, save=False, plot=True, save_figures=False)
    # roi_reconstruct(data_path, cam_calib, (3000, 2000, 3050, 2050), 21, save=False, plot=True, save_figures=False)
    # roi_reconstruct(data_path, cam_calib, (2990, 1990, 2996, 1994), 21, save=True, plot=True, save_figures=True)

    # roi_reconstruct(data_path, cam_calib, (2900, 1600, 3700, 1900), 10, 21, save=True, plot=True, save_figures=False)
    # roi_reconstruct(data_path, cam_calib, (3200, 1700, 3250, 1750), 0, 21, save=False, plot=True, save_figures=False)

    roi_reconstruct(data_path, cam_calib, (4350, 2970, 5600, 3380), 12, 16, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (2420, 3200, 3550, 3800), 8, 12, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (650, 3550, 1950, 3900), 19, 25, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (500, 650, 1500, 1130), 7, 13, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (2600, 350, 3750, 850), 11, 15, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (4650, 650, 5270, 1020), 11, 16, steps=20, ref_id=52, save=True, plot=True, save_figures=True)
    # roi_reconstruct(data_path, cam_calib, (4800, 1700, 5200, 1970), 9, 12, steps=20, ref_id=52, save=True, plot=True, save_figures=True)

    plt.show()
