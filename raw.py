import rawpy
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os.path

# from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

# source: filename or file object
def read_arw(source):
    if type(source) is str:
        with open(source, 'rb') as f:
            raw = rawpy.imread(f)
    else:
        raw = rawpy.imread(source)

    bayer = raw.raw_image_visible.astype(np.float32)
    bayer -= 512
    bayer /= 16372-512
    return np.maximum(0, np.minimum(bayer, 1))

# methods:
#     simple - bilinear with color artifacts (faster custom implementation, corner cases ignored)
#     good   - without color artifacts but adds noise (10 times slower)
#     fancy  - 50 times slower but no artifacts (little noise)
def demosaic(bayer, method="simple"):
    # if method == "good":
    #     return demosaicing_CFA_Bayer_Malvar2004(bayer, pattern='RGGB')
    # elif method == "fancy":
    #     return demosaicing_CFA_Bayer_Menon2007(bayer, pattern='RGGB')
    # elif method != "simple":
    if method != "simple":
        print("Unknown demosaicing method: %s. Use bilinear by default"%(method))

    rgb = np.zeros((bayer.shape[0], bayer.shape[1], 3), dtype=np.float32)

    r = bayer[::2, ::2]
    rgb[::2, ::2, 0] = r
    rgb[::2, 1:-2:2, 0] = 0.5*(r[:, :-1] + r[:, 1:])
    rgb[1:-2:2, ::2, 0] = 0.5*(r[:-1, :] + r[1:, :])
    rgb[1:-2:2, 1:-2:2, 0] = 0.25*(r[:-1, :-1] + r[:-1, 1:] + r[1:, :-1] + r[1:, 1:])

    rgb[:,:,1] = bayer
    rgb[1:-2:2, 1:-2:2, 1] = 0.25*(bayer[1:-2:2, :-2:2] + bayer[1:-2:2, 2::2] + bayer[:-2:2, 1:-2:2] + bayer[2::2, 1:-2:2])
    rgb[2::2, 2::2, 1] = 0.25*(bayer[2::2, 1:-2:2] + bayer[2::2, 3::2] + bayer[1:-2:2, 2::2] + bayer[3::2, 2::2])

    b = bayer[1::2, 1::2]
    rgb[1::2, 1::2, 2] = b
    rgb[1::2, 2::2, 2] = 0.5*(b[:, :-1] + b[:, 1:])
    rgb[2::2, 1::2, 2] = 0.5*(b[:-1, :] + b[1:, :])
    rgb[2::2, 2::2, 2] = 0.25*(b[:-1, :-1] + b[:-1, 1:] + b[1:, :-1] + b[1:, 1:])

    return rgb

# target: raw image of spectrally neutral target (arw filename or rgb in 0-1 range)
# mask_filename: png image with pixels masked red in order to be taken into account
def find_balance(target, mask_filename, plot=True):
    if isinstance(target, str):
        img = demosaic(read_arw(target))[10:4010, 10:6010]
    else:
        img = target

    if mask_filename is None:
        mask = (0.01 < np.min(img, axis=2)) & (np.max(img, axis=2) < 0.9)
    else:
        mask_img = imageio.imread(mask_filename)
        mask = (mask_img[:,:,0] == 255) & (mask_img[:,:,1] == 0) & (mask_img[:,:,2] == 0)

    i, j = np.nonzero(mask)
    slice = img[i, j, :]

    bal = [0,0,0]
    dev = [0,0,0]
    samples=[None, None, None]
    for ch in range(3):
        samples[ch] = slice[:,1]/slice[:,ch]
        mean, std = np.mean(samples[ch]), np.std(samples[ch])
        # print('%s: %f +- %f @ %f %%'%(['r', 'g', 'b'][ch], mean, std, 100*std/mean))
        bal[ch] = mean
        dev[ch] = std

    print("r: %.4f +- %.2f %%\tb: %.4f +- %.2f %%"%(bal[0], 100*dev[0]/bal[0], bal[2], 100*dev[2]/bal[2]))

    if plot:
        plt.figure('target')

        plt.subplot2grid((2, 3), (0, 0))
        plt.gca().set_title('r')
        plt.hist(samples[0], bins=100)

        plt.subplot2grid((2, 3), (1, 0))
        plt.gca().set_title('b')
        plt.hist(samples[2], bins=100)

        plt.subplot2grid((1, 3), (0, 1), colspan=2)
        plt.gca().set_title('masked')
        img[i, j, :] = np.array([1,0,0])
        plt.imshow(img)
        plt.tight_layout()
        plt.show()

    return bal

def balance(rgb, white_balance=None, gamma=None):
    if white_balance is not None:
        if white_balance[0] < 0:
            white_balance = find_balance(rgb, mask_filename=None, plot=False)

        for ch in range(3):
            rgb[:, :, ch] *= white_balance[ch]

    rgb = np.maximum(0, np.minimum(rgb, 1))[2:-2,2:-2,:] # crop corner cases from demosaicing

    if gamma is not None:
        rgb = rgb**(1/gamma) # reverse gamma to linearize

    return rgb

wb_leds = (2.2126, 1, 1.5319)
wb_auto = (-1, -1, -1)
gamma_default = 2.2

IGNORE = 0
OVERWRITE = 1
USE = 2

# caches the result by default
def read_raw(source, cache_raw=IGNORE, white_balance=wb_leds, gamma=gamma_default, demosaicing_method="simple", **kwargs):
    cache = source[:-4] + '.npc' if type(source) is str and cache_raw == USE else None

    if cache and os.path.isfile(cache):
        print('Reuse raw cache:', cache)
        with open(cache, 'rb') as f:
            h, w = 4020, 6020
            img = np.fromfile(f, np.float32, h * w * 3).reshape(h, w, 3)
    else:
        img = balance(demosaic(read_arw(source), method=demosaicing_method), white_balance, gamma)
        cache = source[:-4] + '.npc' if type(source) is str and cache_raw >= OVERWRITE else None
        if cache:
            with open(cache, 'wb') as f:
                img.astype(np.float32).tofile(f)
            print('Saved raw cache:', cache)

    return img

if __name__ == "__main__":
    # plt.figure("simple")
    # plt.imshow(5*read_raw("/home/yurii/data/calib/DSC00090.ARW", use_cache=False, white_balance=(1.5,1,2.6), demosaicing_method="simple"))
    # plt.figure("good")
    # plt.imshow(5*read_raw("/home/yurii/data/calib/DSC00090.ARW", use_cache=False, white_balance=(1.5,1,2.6), demosaicing_method="good"))
    # plt.show()
    # exit(0)

    # data_path = '/home/yurii/data/simple/white_balance/'
    data_path = "D:/paleo-data/1 - FLAT OBJECT 1/"
    target = data_path + 'DSC00094.ARW'
    reference = data_path + 'DSC00094.ARW'
    mask = data_path + 'mask.png'

    wb = find_balance(target, mask)

    plt.figure('reference')
    plt.imshow(read_raw(reference, white_balance=wb))
    plt.show()
