import cv2
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def cuda_loop_slice(img, start, size, steps, plot=False):
    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(cp.int32)
    if plot:
        plt.figure("CUDA Loop Slice Ref")
        plt.imshow(ref.get())

    off_i = start[0] + cp.arange(-steps, steps + 1, dtype=cp.int32)
    off_j = start[1] + cp.arange(-steps, steps + 1, dtype=cp.int32)

    diffs = []
    for p in range(2 * steps + 1):
        diffs.append(cp.average(cp.abs(ref - img[off_i[p]-size:off_i[p]+size+1, off_j[p]-size:off_j[p]+size+1, :])).get())

    if plot:
        plt.figure("CUDA Loop Slice Diffs")
        plt.plot(diffs, ".-")

    return np.array(diffs)


def numpy_loop_slice(img, start, size, steps, plot=False):
    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(np.int32)
    if plot:
        plt.figure("Loop Slice Ref")
        plt.imshow(ref)

    off_i = start[0] + np.arange(-steps, steps + 1, dtype=np.int32)
    off_j = start[1] + np.arange(-steps, steps + 1, dtype=np.int32)

    diffs = []
    for p in range(2 * steps + 1):
        diffs.append(np.average(np.abs(ref - img[off_i[p]-size:off_i[p]+size+1, off_j[p]-size:off_j[p]+size+1, :])))

    if plot:
        plt.figure("Loop Slice Diffs")
        plt.plot(diffs, ".-")

    return np.array(diffs)


def cuda_loop_idx(img, start, size, steps, plot=False):
    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(cp.int32)
    if plot:
        plt.figure("CUDA Loop Idx Ref")
        plt.imshow(ref.get())

    i, j = cp.meshgrid(cp.arange(-size, size+1), cp.arange(-size, size+1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    off_i = start[0] + cp.arange(-steps, steps + 1, dtype=cp.int32)
    off_j = start[1] + cp.arange(-steps, steps + 1, dtype=cp.int32)

    diffs = []
    ref = ref.reshape((-1, 3))
    for p in range(2 * steps + 1):
        diffs.append(cp.average(cp.abs(ref - img[i + off_i[p], j + off_j[p], :])).get())

    if plot:
        plt.figure("CUDA Loop Idx Diffs")
        plt.plot(diffs, ".-")

    return np.array(diffs)


def numpy_loop_idx(img, start, size, steps, plot=False):
    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(np.int32)
    if plot:
        plt.figure("Loop Idx Ref")
        plt.imshow(ref)

    i, j = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    off_i = start[0] + np.arange(-steps, steps + 1, dtype=np.int32)
    off_j = start[1] + np.arange(-steps, steps + 1, dtype=np.int32)

    diffs = []
    ref = ref.reshape((-1, 3))
    for p in range(2 * steps + 1):
        diffs.append(np.average(np.abs(ref - img[i + off_i[p], j + off_j[p], :])))

    if plot:
        plt.figure("Loop Idx Diffs")
        plt.plot(diffs, ".-")

    return np.array(diffs)


def cuda_multi(img, start, size, steps, plot=False):
    # mempool = cp.get_default_memory_pool()
    # mempool.free_all_blocks()

    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(cp.int16)
    if plot:
        plt.figure("CUDA Multi Ref")
        plt.imshow(ref.get())

    i, j = cp.meshgrid(cp.arange(-size, size+1), cp.arange(-size, size+1), indexing="ij")
    i, j = i.astype(cp.int16), j.astype(cp.int16)
    off_i = start[0] + cp.arange(-steps, steps + 1, dtype=cp.int16)
    off_j = start[1] + cp.arange(-steps, steps + 1, dtype=cp.int16)

    # print()
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    I, J = (i[None, :, :] + off_i[:, None, None]).ravel(), (j[None, :, :] + off_j[:, None, None]).ravel()
    # nrefs = cp.repeat(ref[None, ...], 2 * steps + 1, axis=0).reshape((-1, 3))

    # i, j = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1), indexing="ij")
    # i, j = i.astype(np.int16), j.astype(np.int16)
    # off_i = start[0] + np.arange(-steps, steps + 1, dtype=np.int16)
    # off_j = start[1] + np.arange(-steps, steps + 1, dtype=np.int16)
    # I, J = (i[None, :, :] + off_i[:, None, None]).ravel(), (j[None, :, :] + off_j[:, None, None]).ravel()
    # i, J = cp.array(I), cp.array(J)

    # print("\nPrepped I, J")
    # print(I.shape, I.dtype)
    # print(ref.size, I.size, J.size)
    # print(ref.nbytes, I.nbytes, J.nbytes)
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    s = img[I, J, :].reshape((-1, ref.size)).astype(cp.int16)
    # rn = cp.repeat(ref.ravel()[None, :], 2 * steps + 1, axis=0)
    # print("\nSliced")
    # print(s.shape, s.dtype)
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    I, J = None, None
    # print("\nFreed I, J")
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    # rn = ref.ravel()[None, :]
    # d = rn - s
    s -= ref.ravel()[None, :]
    # ad = cp.abs(s)

    # print("\nSubtructed")
    # print(s.shape, s.dtype, s.nbytes)
    # print(ad.shape, ad.dtype)
    # print(ref.size, I.size, J.size, d.size, ad.size)
    # print(ref.nbytes, I.nbytes, J.nbytes, d.nbytes, ad.nbytes)
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    cp.abs(s, out=s)
    # s = None
    # print("\nAbs")
    # print(ad.shape, ad.dtype, ad.nbytes)
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())
    # diffs = cp.average(cp.abs(nrefs - img[I, J, :]).reshape((-1, ref.size)), axis=1)
    diffs = cp.average(s, axis=1)

    # print("Averaged")
    # print(diffs.shape, diffs.dtype)
    # print(ref.size, I.size, J.size, diffs.size)
    # print(ref.nbytes, I.nbytes, J.nbytes, diffs.nbytes)
    # mempool.free_all_blocks()
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())

    if plot:
        plt.figure("CUDA Multi Diffs")
        plt.plot(diffs.get(), ".-")

    return diffs.get()


def numpy_multi(img, start, size, steps, plot=False):
    ref = img[start[0]-size:start[0]+size+1, start[1]-size:start[1]+size+1, :].astype(np.int32)
    if plot:
        plt.figure("Multi Ref")
        plt.imshow(ref)

    i, j = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1), indexing="ij")
    off_i = start[0] + np.arange(-steps, steps + 1, dtype=np.int32)
    off_j = start[1] + np.arange(-steps, steps + 1, dtype=np.int32)

    I, J = (i[None, :, :] + off_i[:, None, None]).ravel(), (j[None, :, :] + off_j[:, None, None]).ravel()
    nrefs = np.repeat(ref[None, ...], 2 * steps + 1, axis=0).reshape((-1, 3))

    diffs = np.average(np.abs(nrefs - img[I, J, :]).reshape((-1, ref.size)), axis=1)

    if plot:
        plt.figure("Multi Diffs")
        plt.plot(diffs, ".-")

    return diffs


if __name__ == "__main__":
    filename = "D:/paleo-data/1 - FLAT OBJECT 1/DSC00084.JPG"
    img = cv2.imread(filename)[:, :, ::-1]

    plt.figure("Img")
    plt.imshow(img)

    start, size, steps, plot = np.array([1900, 1900]), 25, 150, False
    plt.plot([start[0]-steps, start[0] + steps + 1], [start[1] - steps, start[1] + steps + 1], "r-")

    print("\nGPU:")
    # cp.cuda.set_allocator(None)
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=2*1024**3)

    cimg = cp.asarray(img)#.astype(cp.int16)
    # cuda_loop_slice(cimg, cp.asarray(start), size, steps, plot=plot)
    # t0 = time.time()
    # ret_g1 = cuda_loop_slice(cimg, cp.asarray(start), size, steps, plot=plot)
    # print("Slice", time.time() - t0)
    #
    # cuda_loop_idx(cimg, cp.asarray(start), size, steps, plot=plot)
    # t0 = time.time()
    # ret_g2 = cuda_loop_idx(cimg, cp.asarray(start), size, steps, plot=plot)
    # print("Idx  ", time.time() - t0)

    # cuda_multi(cimg, cp.asarray(start), size, steps, plot=plot)
    cuda_multi(cimg, start, size, steps, plot=plot)

    print("\n")

    t0 = time.time()
    for i in range(10000):
        ret_g3 = cuda_multi(cimg, start, size, steps, plot=plot)
    print("Multi", time.time() - t0)

    print(mempool.used_bytes())
    print(mempool.total_bytes())

    print("\nCPU:")

    t0 = time.time()
    for i in range(1000):
        ret_c1 = numpy_loop_slice(img, start, size, steps, plot=plot)
    print("Slice", time.time() - t0)

    # t0 = time.time()
    # ret_c2 = numpy_loop_idx(img, start, size, steps, plot=plot)
    # print("Idx  ", time.time() - t0)
    #
    # t0 = time.time()
    # ret_c3 = numpy_multi(img, start, size, steps, plot=plot)
    # print("Multi", time.time() - t0)

    # assert np.array_equal(ret_g1, ret_c1)
    # assert np.array_equal(ret_g2, ret_c2)
    # assert np.array_equal(ret_g3, ret_c3)
    # assert np.array_equal(ret_c1, ret_c2)
    # assert np.array_equal(ret_c2, ret_c3)
    assert np.array_equal(ret_c1, ret_g3)

    plt.show()