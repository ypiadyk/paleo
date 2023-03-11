from utils import *
from calibrate import *


def calibrate_intrinsic(data_path, max_images=70, min_points=80, centerPrincipalPoint=None, save=False, plot=False, **kw):
    corners = load_corners(data_path + "detected/corners.json")
    names = [k for k, v in corners.items()]
    imgs = [v["img"] for k, v in corners.items()]
    objs = [v["obj"] for k, v in corners.items()]
    print("Detected:", len(objs))

    img, name, points = cv2.imread(data_path + names[0]), names[0], imgs[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    idx = [i for i in range(len(imgs)) if imgs[i].shape[0] > min_points]
    print("Has enough points (>%d):" % min_points, len(idx))

    names, imgs, objs = [names[i] for i in idx], [imgs[i] for i in idx], [objs[i] for i in idx]

    if len(imgs) > max_images:
        stride = len(imgs) // max_images + 1
        names, imgs, objs = names[::stride], imgs[::stride], objs[::stride]

    if save:
        ensure_exists(data_path + "calibrated/")

    calibration, errors = calibrate(objs, imgs, gray.shape, centerPrincipalPoint=centerPrincipalPoint,
                                                            out_dir=data_path + "calibrated/", plot=plot, save_figures=save, **kw)
    mtx, dist, new_mtx, roi = calibration

    if plot or save:
        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
        u_points = cv2.undistortPoints(points, mtx, dist, None, new_mtx).reshape(-1, 2)

    if plot:
        all_errors = []
        for error in errors[1][1]:
            all_errors.extend(error)

        plt.figure("All errors", (12, 8))
        plt.clf()
        plt.hist(all_errors, bins=50, range=[0, 2])
        plt.xlabel("Error, pixels")
        plt.ylabel("Counts")
        plt.xlim([0, 2])
        plt.tight_layout()

        if save:
            plt.savefig(data_path + "calibrated/all_camera_reprojection_errors.png", dpi=300)

        def plot_rec(p):
            plt.plot(p[:, 0], p[:, 1], ".g")
            tl, tr = np.argmin(p[:, 0] + p[:, 1]), np.argmin(-p[:, 0] + p[:, 1])
            bl, br = np.argmax(-p[:, 0] + p[:, 1]), np.argmax(p[:, 0] + p[:, 1])
            plt.plot(p[[tl, tr, br, bl, tl], 0], p[[tl, tr, br, bl, tl], 1], "-r")
            return [tl, tr, br, bl, tl]

        def draw_rec(img, p, idx):
            for i in range(len(idx)-1):
                img = cv2.line(img, tuple(p[idx[i], :].astype(np.int)), tuple(p[idx[i+1], :].astype(np.int)), (0, 0, 255), thickness=2)
            return img

        def draw_points(img, p):
            for i in range(p.shape[0]):
                img = cv2.circle(img, tuple(p[i, :].astype(np.int)), 10, (0, 255, 0), thickness=2)
            return img

        plt.figure("Original", (12, 9))
        plt.clf()
        plt.imshow(img[:, :, ::-1])
        plt.plot(mtx[0, 2], mtx[1, 2], ".b")
        idx = plot_rec(points)
        plt.tight_layout()

        img = draw_rec(img, points, idx)
        img = draw_points(img, points)
        img = cv2.circle(img, tuple(mtx[:2, 2].astype(np.int)), 5, (255, 0, 0), thickness=10)

        plt.figure("Undistorted", (12, 9))
        plt.clf()
        plt.imshow(undistorted[:, :, ::-1])
        plt.plot(new_mtx[0, 2], new_mtx[1, 2], ".b")

        idx = plot_rec(u_points)
        plt.tight_layout()

        undistorted = draw_rec(undistorted, u_points, idx)
        undistorted = draw_points(undistorted, u_points)
        undistorted = cv2.circle(undistorted, tuple(new_mtx[:2, 2].astype(np.int)), 5, (255, 0, 0), thickness=10)

    if save:
        save_camera_calibration(calibration, data_path + "calibrated/geometry.json", mean_error=np.mean(errors[1][0]))
        cv2.imwrite(data_path + "calibrated/" + name[:-4] + '_original.png', img)
        cv2.imwrite(data_path + "calibrated/" + name[:-4] + '_undistorted.png', undistorted)

    return calibration, errors[1]


def save_camera_calibration(calibration, filename, mean_error=None):
    with open(filename, "w") as f:
        json.dump({"mtx": calibration[0],
                   "dist": calibration[1],
                   "new_mtx": calibration[2],
                   "roi": calibration[3],
                   "mean_projection_error, pixels": mean_error}, f, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    # data_path = "D:/paleo-data/CALIBRATION BOARD 1e2/"
    # data_path = "D:/paleo-data/CALIBRATION BOARD 3e4/"
    data_path = "D:/paleo-data/CALIBRATION BOARD 5/"

    calibration, errors = calibrate_intrinsic(data_path, error_thr=1.1, save=True, plot=True)

    plt.show()