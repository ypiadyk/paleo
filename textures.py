from utils import *
from natsort import natsorted
from image_similarity_measures.evaluate import evaluation
from image_similarity_measures.quality_metrics import rmse, fsim, issm, ssim, psnr, sam, sre, uiq

import scipy
import scipy.cluster.hierarchy as sch


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    # if not inplace:
    #     corr_array = corr_array.copy()

    return corr_array[idx, :][:, idx], idx


def dist(crop1, crop2, method="L2"):
    if method == "L2":
        diff = (crop1 - crop2).ravel()
        return np.sqrt(np.average(diff*diff) / (diff.shape[0]-1))
    elif method == "RMSE":
        return rmse(crop1, crop2)
    elif method == "ISSM":
        return issm(crop1, crop2)  # All zeros
    elif method == "SSIM":
        return ssim(crop1, crop2)
    elif method == "FSIM":
        return fsim(crop1, crop2)  # Slow
    elif method == "PSNR":
        return psnr(crop1, crop2)  # Has NaNs
    elif method == "SAM":
        return sam(crop1, crop2)
    elif method == "SRE":
        return sre(crop1, crop2)  # Has NaNs
    elif method == "UIQ":
        return uiq(crop1, crop2)  # Very slow
    else:
        return method(crop1, crop2)


def crop():
    in_dir = "D:/Dropbox/work/paleo/fossil-textures/crops_organized/"
    out_dir = "D:/Dropbox/work/paleo/fossil-textures/crops_all_adjusted/"

    groups = glob.glob(in_dir + "*.BMP")
    w, h, low, high = 10000, 10000, 255, 0
    fake_scale = 1.0

    for group in groups:
        images = glob.glob(group[:-4] + "/*.bmp")
        for image in images:
            img = cv2.imread(image)
            if "fake" in image:
                print(image, img.shape, img.dtype, np.min(img), np.max(img))
                img = (fake_scale * img.astype(np.float32)).astype(np.uint8)
                # continue

            w, h = min(w, img.shape[1]), min(h, img.shape[0])
            low, high = min(low, np.min(img)), max(high, np.max(img))

    print(w, h, low, high)

    for group in groups:
        images = glob.glob(group[:-4] + "/*.bmp")
        for image in images:
            img = cv2.imread(image)
            if "fake" in image:
                img = (fake_scale * img.astype(np.float32)).astype(np.uint8)

            dirname = group[group.rfind("\\")+1:-4]
            filename = os.path.basename(image)
            # print(group, dirname, filename)

            save_to = out_dir + dirname + "_" + filename
            # print(save_to)

            l, t = (img.shape[1] - w) // 2, (img.shape[0] - h) // 2
            img = img[t:t+h-1, l:l+w-1, :]
            img = 255.0 * (img.astype(np.float32) - low) / (high - low)
            cv2.imwrite(save_to, img.astype(np.uint8))

    exit(0)


if __name__ == "__main__":
    # crop()
    data_path = "D:/Dropbox/work/paleo/fossil-textures/crops_all_adjusted/"

    files = natsorted(glob.glob(data_path + "*.bmp"))
    crops = [cv2.imread(file) for file in files]

    files = [os.path.basename(file) for file in files]
    crops = np.stack(crops).astype(np.float32)

    print(files)
    print(crops.shape, crops.dtype)

    # for method in ["L2", "RMSE", "ISSM"]:
    for method in ["RMSE", "SSIM"]:
    # for method in ["PSNR"]:
    # for method in ["SAM"]:
    # for method in ["FSIM"]:
    # for method in ["SRE"]:
        print("\n", method)
        n = crops.shape[0]
        # n = 30
        corr = np.zeros((n, n))
        for i in range(n):
            print(i)

            # for j in range(n):
            #     corr[i, j] = dist(crops[i, :, :, :], crops[j, :, :, :], method=method)

            jobs = [joblib.delayed(dist)(crops[i, :, :, :], crops[j, :, :, :], method=method) for j in range(n)]
            results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=1, pre_dispatch="all")(jobs)
            corr[i, :] = np.array(results)


        if method in ["L2", "RMSE"]:
            m = np.max(corr)
            corr = 1 - corr/m
        elif method in ["SSIM"]:
            m = np.min(corr)
            corr = (corr - m) / (1 - m)

        if method in ["L2", "RMSE", "SSIM"]:
            corr, idx = cluster_corr(corr)
            files_idx = [files[i] for i in idx]
            # files_idx = files
        else:
            files_idx = files

        title = "Correlation Matrix - " + method
        plt.figure(title, (16, 13))
        plt.clf()
        plt.imshow(corr)
        plt.colorbar()
        skip, sz = 1, 6
        plt.xticks(np.arange(n)[::skip], files_idx[:n][::skip], fontsize=sz, rotation=90)
        plt.yticks(np.arange(n)[::skip], files_idx[:n][::skip], fontsize=sz)
        plt.title(title, fontsize=2*sz)
        plt.tight_layout()
        plt.savefig(title + ".png", dpi=300)

    plt.show()
