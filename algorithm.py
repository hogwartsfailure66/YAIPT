import numpy as np
from numpy import ndarray


def isGrayScale(img: ndarray):
    if img.ndim == 2:
        return True
    return False


def plane(img: ndarray) -> ndarray:
    if isGrayScale(img):
        h, w = img.shape
        # print(h, w)
        # print(img)
        tmp = np.ones((3 * h, 3 * w), dtype=np.uint8)
        for i in range(8):
            col, row = i % 3, i // 3
            # print(i, row, col)
            p = 7 - i
            for j in range(h):
                for k in range(w):
                    tmp[row * h + j][col * w + k] = (img[j][k] >> p) & 0b1
        # print(tmp)
        return tmp
    print("error! not a gray scale image")
    return None


def myEqualize(img: ndarray):
    hist, bins = np.histogram(img.ravel(), 256)
    r = hist.cumsum()
    r = r / r[-1]
    tmp = np.interp(img.ravel(), bins[:-1], r)
    return tmp.reshape(img.shape)


def equalize(img: ndarray) -> ndarray:
    if isGrayScale(img):
        return myEqualize(img)
    else:  # rgb
        for i in range(3):
            img[:, :, i] = myEqualize(img[:, :, i])
        return img


def myDenoise(img: ndarray):
    h, w = img.shape
    tmp = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # print(i,j)
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                tmp[i][j] = img[i][j]
            else:
                sum = 0
                for m in range(i - 1, i + 2):
                    for n in range(j - 1, j + 2):
                        sum += img[m][n]
                tmp[i][j] = 1 / 9 * sum
    return tmp.reshape(img.shape)


def denoise(img: ndarray) -> ndarray:
    if isGrayScale(img):
        return myDenoise(img)
    else:
        for i in range(3):
            img[:, :, i] = myEqualize(img[:, :, i])
        return img


def myNearestInterpolate(img: ndarray):
    # 最近邻
    h, w = img.shape
    height, width = h * 2, w * 2
    tmp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            m = round(i / height * h)
            n = round(j / width * w)
            if m == h:
                m = h - 1
            if n == w:
                n = w - 1
            tmp[i, j] = img[m, n]
    return tmp.reshape((height, width))


def myInterpolate(img: ndarray):
    # 双线性 写的太拉，有时间再改
    h, w = img.shape
    # print(h, w)
    height, width = 2 * h, 2 * w
    tmp = np.zeros((height, width))
    for i in range(h):
        for j in range(w):
            tmp[2 * i][2 * j] = img[i][j]
    for i in range(1, height, 2):
        for j in range(0, width, 2):
            if i < height - 1:
                tmp[i][j] = (tmp[i - 1][j] + tmp[i + 1][j]) / 2
            else:
                tmp[i][j] = tmp[i - 1][j]
    for i in range(0, height, 2):
        for j in range(1, width, 2):
            if j < width - 1:
                tmp[i][j] = (tmp[i][j - 1] + tmp[i][j + 1]) / 2
            else:
                tmp[i][j] = tmp[i][j - 1]
    for i in range(1, height, 2):
        for j in range(1, width, 2):
            if i == height - 1 and j == width - 1:
                tmp[i][j] = (tmp[i - 1][j] + tmp[i][j - 1]) / 2
            elif i == height - 1:
                tmp[i][j] = (tmp[i][j - 1] + tmp[i][j + 1] + tmp[i - 1][j]) / 3
            elif j == width - 1:
                tmp[i][j] = (tmp[i][j - 1] + tmp[i + 1][j] + tmp[i - 1][j]) / 3
            else:
                tmp[i][j] = (tmp[i][j - 1] + tmp[i + 1][j] + tmp[i - 1][j] + tmp[i][j + 1]) / 4
    # print(tmp)
    return tmp.reshape((height, width))


def interpolate(img: ndarray) -> ndarray:
    if isGrayScale(img):
        # return myNearestInterpolate(img)
        return myInterpolate(img)
    else:
        h, w, d = img.shape
        height, width = h * 2, w * 2
        tmp = np.zeros((height, width, d))
        for i in range(3):
            # tmp[:, :, i] = myNearestInterpolate(img[:, :, i])
            tmp[:, :, i] = myInterpolate(img[:, :, i])
        return tmp


def dft(img: ndarray) -> ndarray:
    if isGrayScale(img):
        tmp = np.fft.fft2(img)
        tmp = np.fft.fftshift(tmp)
        tmp = np.abs(tmp)
        return np.log(tmp)
    print("error! not a gray scale image")
    return None


def butterworth(img: ndarray) -> ndarray:
    if isGrayScale(img):
        h, w = img.shape
        halfX = h // 2
        halfY = w // 2
        d0 = 40
        n = 2
        nTimes2 = 2 * n
        tmp = np.fft.fft2(img)
        tmp = np.fft.fftshift(tmp)
        m = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                d = np.sqrt((i - halfX) ** 2 + (j - halfY) ** 2)
                m[i, j] = 1 / (1 + (d / d0) ** nTimes2)
        ret = np.fft.ifftshift(tmp * m)
        ret = np.fft.ifft2(ret)
        ret = np.abs(ret)
        return ret
    print("error! not a gray scale image")
    return None


def canny(img: ndarray) -> ndarray:
    return None


def myErode(img: ndarray):
    h, w = img.shape
    ret = img.copy()
    m = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
    tmp = np.pad(img, 1)
    # print(tmp)
    for i in range(1, h):
        for j in range(1, w):
            if np.sum(m * tmp[i - 1:i + 2, j - 1:j + 2]) < 255 * 4:
                ret[i, j] = 0
    # print(ret)
    return ret


def myInflate(img: ndarray):
    h, w = img.shape
    ret = img.copy()
    m = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
    tmp = np.pad(img, 1)
    for i in range(1, h):
        for j in range(1, w):
            if np.sum(m * tmp[i - 1:i + 2, j - 1:j + 2]) >= 255:
                ret[i, j] = 255
    return ret


def morphology(img: ndarray) -> ndarray:
    if isGrayScale(img):
        if img.dtype == np.float:
            img = (img * 255).astype(np.uint8)
        tmp = myErode(img)
        tmp = myInflate(tmp)
        return tmp
    print("error! not a gray scale image")
    return None
