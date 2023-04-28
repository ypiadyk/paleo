import matcher_module
import numpy as np


def compute_diffs(r, w, img, ms, pad):
    diffs = []
    for mi in range(ms.shape[0]):
        patch = img[ms[mi, 1] - pad:ms[mi, 1] + pad + 1, ms[mi, 0] - pad:ms[mi, 0] + pad + 1, :].astype(np.float32)
        loss = w[:, :, None] * ((r - patch) ** 2)
        diffs.append(np.average(loss))

    return np.array(diffs, dtype=np.float32)


if __name__ == '__main__':
    print("Hi")
    print(matcher_module.in_bounds_func(0, 1, 100, 100))
    print(matcher_module.in_bounds_func(0, 200, 100, 100))

    n, pad = 100, 20
    # r = np.zeros((2*pad+1, 2*pad+1, 3), dtype=np.uint8)
    r = np.random.randint(255, size=(2*pad+1, 2*pad+1, 3)).astype(np.uint8)
    # w = np.ones((2*pad+1, 2*pad+1), dtype=np.float32)
    w = np.random.rand(2*pad+1, 2*pad+1).astype(np.float32)
    # img = np.ones((1000, 1000, 3), dtype=np.uint8)
    img = np.random.randint(255, size=(1000, 1000, 3)).astype(np.uint8)
    ms = 100 * np.ones((n, 2), dtype=np.int32)
    diffs = np.zeros(n, dtype=np.float32)

    matcher_module.compute_diffs(r, w, img, ms, pad, diffs)
    print(diffs)

    ref_diffs = compute_diffs(r, w, img, ms, pad)
    print(ref_diffs)
    print()
    print(ref_diffs - diffs)

    exit(0)
    mask = np.zeros((10, 10), dtype=np.uint32)
    mask[:3, :3] = 10
    mask[7:, 7:] = 15
    mask[5, :] = 10

    data = np.zeros((10, 10), dtype=np.double)
    data[:, :5] = 1
    data[:, 5:] = 10

    print(mask, "\n")
    print(data, "\n")

    matcher_module.label(mask, data, 5)

    print(mask, "\n")
    print(data, "\n")

