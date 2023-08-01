import joblib
from utils import *
from calibrate import *
from raw import *


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
        plt.hist(all_errors, bins=100, range=[0, 3])
        plt.xlabel("Error, pixels")
        plt.ylabel("Counts")
        plt.xlim([0, 3])
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
                img = cv2.line(img, tuple(p[idx[i], :].astype(np.int32)), tuple(p[idx[i+1], :].astype(np.int32)), (0, 0, 255), thickness=2)
            return img

        def draw_points(img, p):
            for i in range(p.shape[0]):
                img = cv2.circle(img, tuple(p[i, :].astype(np.int32)), 10, (0, 255, 0), thickness=2)
            return img

        plt.figure("Original", (12, 9))
        plt.clf()
        plt.imshow(img[:, :, ::-1])
        plt.plot(mtx[0, 2], mtx[1, 2], ".b")
        idx = plot_rec(points)
        plt.tight_layout()

        img = draw_rec(img, points, idx)
        img = draw_points(img, points)
        img = cv2.circle(img, tuple(mtx[:2, 2].astype(np.int32)), 5, (255, 0, 0), thickness=10)

        plt.figure("Undistorted", (12, 9))
        plt.clf()
        plt.imshow(undistorted[:, :, ::-1])
        plt.plot(new_mtx[0, 2], new_mtx[1, 2], ".b")

        idx = plot_rec(u_points)
        plt.tight_layout()

        undistorted = draw_rec(undistorted, u_points, idx)
        undistorted = draw_points(undistorted, u_points)
        undistorted = cv2.circle(undistorted, tuple(new_mtx[:2, 2].astype(np.int32)), 5, (255, 0, 0), thickness=10)

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


def undistort_images(images_path, cam_calib):
    images = glob.glob(images_path + "*.jpg")
    ensure_exists(images_path + "undistorted/")
    print("Found %d images:" % len(images), images)

    def undistort_single(image):
        original = cv2.imread(image)
        undistorted = cv2.undistort(original, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])
        new_filename = images_path + "undistorted/" + os.path.basename(image)
        # cv2.imwrite(new_filename, undistorted)

        wb = load_wb(images_path + "white_balance/white_balance.json")
        arw = read_raw(image[:-4] + ".ARW", white_balance=wb)[10:4010, 10:6010]
        arw = (255*arw).astype(np.uint8)
        u_arw = cv2.undistort(arw, cam_calib["mtx"], cam_calib["dist"], newCameraMatrix=cam_calib["new_mtx"])
        cv2.imwrite(new_filename[:-4] + ".bmp", u_arw[:, :, ::-1])
        # print(new_filename)

    jobs = [joblib.delayed(undistort_single)(image) for image in images]
    joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)


if __name__ == "__main__":
    # calib_data = "D:/Dropbox/work/cvpr/1_calib_frames/"
    # calib_data = "D:/Dropbox/work/cvpr/2_calib_frames/"
    # calib_data = "D:/Dropbox/work/cvpr/3_calib_frames/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 1e2/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 3e4/"
    # calib_data = "D:/paleo-data/CALIBRATION BOARD 5/"
    # calib_data = "D:/paleo-data/test-calib/"
    # calib_data = "D:/paleo_scans/calib/"
    # calib_data = "D:/paleo_scans/calib_11/"
    calib_data = "D:/paleo_scans/calib_17/"

    calibration, errors = calibrate_intrinsic(calib_data, error_thr=18.1, min_points=20, save=True, plot=True)
    # cam_calib = load_calibration(calib_data + "/calibrated/geometry.json")

    # image_data = "D:/Dropbox/work/cvpr/1_fscam_frames/"
    # image_data = "D:/Dropbox/work/cvpr/2_fscam_frames/"
    # image_data = "D:/Dropbox/work/cvpr/3_fscam_frames/"
    # image_data = "D:/paleo-data/1 - FLAT OBJECT 1/"
    # image_data = "D:/paleo-data/2 - FLAT OBJECT 2/"
    # image_data = "D:/paleo-data/3 - IRREGULAR OBJECT 1/"
    # image_data = "D:/paleo-data/4 - IRREGULAR OBJECT 2/"
    # image_data = "D:/paleo-data/5 - BOX/"
    # image_data = "D:/paleo-data/test-scan-1/"
    # image_data = "D:/paleo-data/test-scan-2/"

    # image_data = "D:/paleo_scans/scan_0/"
    # image_data = "D:/paleo_scans/scan_1/"
    # image_data = "D:/paleo_scans/scan_2/"
    image_data = "D:/paleo_scans/scan_17/"

    # undistort_images(image_data, cam_calib)

    plt.show()
