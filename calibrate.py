import json
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from detect import load_corners


def fit_line(points):
    center = np.mean(points, axis=0)
    uu, dd, vv = np.linalg.svd(points - center)
    return center, vv[0]


def point_line_dist(p, l0, l1):
    return np.linalg.norm(np.cross(l1 - l0, p - l0)) / np.linalg.norm(l1 - l0)


def projection_errors(obj_points, img_points, calibration):
    ret, mtx, dist, rvecs, tvecs = calibration
    center = mtx[:2, 2]

    avg_errors, all_errors, projections, distances = [], [], [], []
    for i, (obj_p, img_p) in enumerate(zip(obj_points, img_points)):
        img_points_2, _ = cv2.projectPoints(obj_p, rvecs[i], tvecs[i], mtx, dist)
        img_points_2 = img_points_2.reshape((obj_p.shape[0], 2))
        all_errors.append(np.linalg.norm(img_p - img_points_2, axis=1))
        avg_errors.append(np.average(all_errors[-1]))
        projections.append(img_points_2)
        distances.append(np.linalg.norm(center - img_points_2, axis=1))

    return np.array(avg_errors), all_errors, projections, distances


def calibrate(obj_points, img_points, dim, error_thr=1.0, mtx_guess=None, no_tangent=False, less_K=False,
                                centerPrincipalPoint=None, out_dir="", plot=False, save_figures=True, **kw):
    h, w, n = dim[0], dim[1], len(img_points)
    print("Initial:", n)

    flags = cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_K3
    if mtx_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    initial_calibration = cv2.calibrateCamera(obj_points, img_points, (w, h), mtx_guess, None, flags=flags)
    mtx_guess = initial_calibration[1]

    flags = cv2.CALIB_FIX_TANGENT_DIST if no_tangent else 0
    if less_K:
        flags |= cv2.CALIB_FIX_K3
    if mtx_guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    initial_errors = projection_errors(obj_points, img_points, initial_calibration)
    print("Mean initial error:", np.mean(initial_errors[0]))

    selected = np.nonzero(initial_errors[0] < error_thr)[0]
    print("Selected:", len(selected))

    obj_selected, img_selected = [obj_points[i] for i in selected], [img_points[i] for i in selected]
    refined_calibration = cv2.calibrateCamera(obj_selected, img_selected, (w, h), mtx_guess, None, flags=flags)

    refined_errors = projection_errors(obj_selected, img_selected, refined_calibration)
    print("Mean selected error:", np.mean(refined_errors[0]))

    ret, mtx, dist, rvecs, tvecs = refined_calibration
    print("\nmtx:\n", mtx)
    print("\ndist:", dist)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.33, (w, h), centerPrincipalPoint=centerPrincipalPoint)
    print("\nnew_mtx:\n", new_mtx)
    print("\nroi:", roi)

    if plot:
        plt.figure("Calibration", (12, 8))
        plt.clf()
        plt.plot(np.arange(n), initial_errors[0], ".r", markersize=3.5, label="Initial Errors")
        plt.plot(selected, refined_errors[0], ".b", markersize=3.5, label="Selected Images")
        plt.plot([-1, n], [error_thr, error_thr], '--k', linewidth=1.25, label="Threshold (%.1f)" % error_thr)
        plt.title("Images Selection")
        plt.xlabel("Image, #")
        plt.ylabel("Error, pixels")
        plt.xlim([-2, n+1])
        plt.ylim([0, 1.1 * np.max(initial_errors[0])])
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig(out_dir + "/images_selection_criteria.png", dpi=300)

    calibration = mtx, dist.ravel(), new_mtx, np.array(roi)
    errors = initial_errors, refined_errors, selected

    return calibration, errors


def trace_ray(T, R, p, d):
    A = np.stack((R[:, 0], R[:, 1], -d), axis=1)
    b = p - T
    uvt = np.matmul(np.linalg.inv(A), b)
    return p + uvt[2]*d


def lift_to_3d(p_img, mtx, T, R, offset=0.0):
    p_world = np.zeros((p_img.shape[0], 3))
    for i in range(p_img.shape[0]):
        p_world[i, :] = trace_ray(T + offset * R[:, 2], R, np.zeros((3)), np.array([(p_img[i, 0] - mtx[0, 2]) / mtx[0, 0],
                                                                                    (p_img[i, 1] - mtx[1, 2]) / mtx[1, 1], 1]))
    return p_world


def reconstruct_planes(data_path, camera_calib, min_points=10, undistorted=False):
    cam_mtx, cam_dist, cam_new_mtx = camera_calib["mtx"], camera_calib["dist"], camera_calib["new_mtx"]
    charuco = load_corners(data_path + "/undistorted/detected/corners.json")

    charuco_template = "blank_%d.png"
    charuco_img, charuco_2d, charuco_3d, charuco_id, charuco_frame = [], [], [], [], []

    avg_errors, all_errors = [], []
    for name in sorted(charuco.keys()):
        c_obj = charuco[name]["obj"]
        c_idx = charuco[name]["idx"]
        c_img = charuco[name]["img"].astype(np.float32).reshape((-1, 2))

        if not undistorted:
            c_img = cv2.undistortPoints(c_img.reshape((-1, 1, 2)), cam_mtx, cam_dist, P=cam_new_mtx).reshape((-1, 2))

        if len(c_idx) < min_points:
            continue

        ret, rvec, tvec = cv2.solvePnP(c_obj, c_img, cam_new_mtx, None)
        T, (R, _) = tvec.ravel(), cv2.Rodrigues(rvec)

        projected = cv2.projectPoints(c_obj, rvec, tvec, cam_new_mtx, None)[0].reshape(-1, 2)
        all_errors.append(np.linalg.norm(c_img - projected, axis=1))
        avg_errors.append(np.average(all_errors[-1]))

        c_3d = np.matmul(R, c_obj.T) + tvec
        charuco_img.append(c_img)
        charuco_2d.append(c_obj)
        charuco_3d.append(c_3d.T)
        charuco_id.append(c_idx)
        charuco_frame.append((T, R))

    chrk = (charuco_img, charuco_2d, charuco_3d, charuco_id, charuco_frame)

    return chrk, (avg_errors, all_errors)
