import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from utils import *
from calibrate import *


def calibrate_extrinsic(data_path, cam_calib, min_plane_points=10, save=None, plot=False, save_figures=None, **kw):
    save = save or False
    save_figures = save_figures or save

    charuco, charuco_errors = reconstruct_planes(data_path, cam_calib, min_points=min_plane_points, undistorted=True)

    charuco_img, charuco_2d, charuco_3d, charuco_id, charuco_frame = charuco
    print("\nReconstructed planes:", len(charuco_3d))
    print("Mean plane error", np.mean(charuco_errors[0]))

    all_points, all_ids = np.concatenate(charuco_3d), np.concatenate(charuco_id)
    max_id = np.max(all_ids)
    # c_centers, c_errors, p_errors = [], [], []
    print("Max corner id:", max_id)

    if save or save_figures:
        data_path += "/reconstructed/"
        ensure_exists(data_path)

    if plot:
        plt.figure("Cameras Reconstruction", (11, 10))
        plt.clf()
        ax = plt.subplot(111, projection='3d', proj_type='ortho')
        # ax.set_title("Cameras Reconstruction")

        board(ax, np.zeros(3), np.eye(3), label="Charuco Board")

        O, B = [], []
        for i in range(len(charuco_frame)):
            Ti, Ri = charuco_frame[i]
            o, b = np.matmul(Ri.T, -Ti), Ri.T
            basis(ax, o, b)
            O.append(o)
            B.append(b)
            # board(ax, Ti, Ri, label="Charuco Boards" if i == 0 else "")

        O = np.array(O)
        plt.plot(O[:, 0], O[:, 1], O[:, 2], "m--.", label="Camera Path")

        scatter(ax, np.concatenate(charuco_2d, axis=0), c="g", s=4, label="Detected Corners")
        # scatter(ax, np.concatenate(charuco_3d, axis=0), c="g", s=4, label="Detected Corners")
        # scatter(ax, np.array(c_centers), c="r", s=16, label="Circle Centers")
        # line(ax, p - 200 * dir, p + 225 * dir, "-b", label="Stage Axis")

        ax.set_xlabel("x, mm")
        ax.set_ylabel("y, mm")
        ax.set_zlabel("z, mm")
        plt.legend()
        plt.tight_layout()
        axis_equal_3d(ax, zoom=1.2)

        if save_figures:
            ax.view_init(azim=113, elev=27)
            plt.savefig(data_path + "/reconstruction_view.png", dpi=320)

        # plt.figure("Errors", (7, 3.2))
        # plt.clf()
        # # plt.subplot(1, 3, 1, title="Camera reprojection")
        # # plt.hist(np.concatenate(charuco_errors[1]), bins=50)
        # # plt.xlabel("Error, pixels")
        # # plt.tight_layout()
        #
        # plt.subplot(1, 2, 1)#, title="Circle Fit")
        # plt.title("Circle Fit", fontsize=11)
        # plt.hist(c_errors, bins=40, range=[-0.3, 0.3])
        # plt.xlim([-0.3, 0.3])
        # plt.xlabel("Error, mm")
        # plt.ylabel("Counts")
        # plt.tight_layout()
        #
        # plt.subplot(1, 2, 2)#, title="Axis Fit")
        # plt.title("Axis Fit", fontsize=11)
        # plt.hist(axis_errors, bins=40, range=[0, 0.2])
        # plt.xlim([0, 0.2])
        # plt.xlabel("Error, mm")
        # plt.ylabel("Counts")
        # plt.tight_layout()

        # if save_figures:
        #     plt.savefig(data_path + "/stage_errors.png", dpi=300)

    if save:
        with open(data_path + "/extrinsic.json", "w") as f:
            json.dump({"O": O,
                       "B": B}, f, indent=4, cls=NumpyEncoder)

    # return p, dir, axis_errors


if __name__ == "__main__":
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 1e2/"
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

    calibrate_extrinsic(data_path, cam_calib, save=True, plot=True, save_figures=True)

    plt.show()

