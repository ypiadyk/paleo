from detect import *
from intrinsic import *
from extrinsic import *
from reconstruct import *


if __name__ == "__main__":
    calib_data = "D:/paleo_scans/calib_jpg/"
    scan_data = "D:/paleo_scans/scan_raw/"
    plot = True  # Choose whether to plot intermediate results

    # Define calibration board used for actual camera calibration
    calib_n, calib_m, calib_size = 28, 17, 20  # mm

    # Detect markers for the calibration images (comment after done)
    detect_all(calib_data + "*.JPG", detect_charuco, n=calib_n, m=calib_m, size=calib_size,
               save=True, draw=True, pre_scale=2, draw_scale=1)

    # Compute the actual camera calibration (comment first line when done)
    calibration, errors = calibrate_intrinsic(calib_data, error_thr=18.1, min_points=20, save=True, plot=plot)
    cam_calib = load_calibration(calib_data + "/calibrated/geometry.json")

    # Define target image and the white square mask for the while balance computation (comment when done)
    target = scan_data + 'white_balance/target.arw'  # Copy a raw image containing the color calibration target
    mask = scan_data + 'white_balance/mask.png'  # Draw red-on-white mask for white square in 2019 Edition corner
    find_balance(target, mask, save=True, plot=plot)

    # Undistort and color adjust the raw images in preparation for reconstruction (comment after done)
    undistort_images(scan_data, cam_calib)

    # Define calibration board used during fossil scanning
    scan_n, scan_m, scan_size = 25, 18, 30  # mm

    # Detect markers fopm scan images for extrinsic calibration (comment after done)
    detect_all(scan_data + "undistorted/*.bmp", detect_charuco, n=scan_n, m=scan_m, size=scan_size,
               save=True, draw=True, pre_scale=2, draw_scale=1)

    # Performs the actual extrinsic calibration (comment when done)
    calibrate_extrinsic(scan_data, cam_calib, save=True, plot=plot, save_figures=plot)

    # Define search range(s) and max steps count (per mm)
    # min_h, max_h, steps = 00, 50, 4  # mm / per mm
    min_h, max_h, steps = 19, 24, 20  # mm / per mm
    ref_id = 52  # Select (counting from zero) the most centered reference image for roi definition
    roi = (2200, 1600, 3900, 2300)  # Reconstruction roi (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    # Perform the actual reconstruction (caches patch matching data on per range basis but not ref_id or roi)
    roi_reconstruct(scan_data, cam_calib, roi, min_h, max_h, steps=steps, ref_id=ref_id,
                    save=True, plot=plot, save_figures=plot)

    # Shows (a ton of) plots if selected
    if plot:
        plt.show()
