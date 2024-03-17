import os
import cv2
import numpy as np
from PIL import Image

# 高斯滤波器
def gaussianKernel(k, sigma):
    kernel_h = np.zeros((2 * k + 1, 2 * k + 1))
    for i in range(0, 2 * k + 1):
        for j in range(0, 2 * k + 1):
            X, Y = i - k, j - k
            kernel_h[i][j] = np.exp(- (X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel_h /= np.sum(kernel_h)
    return kernel_h


# 二维卷积
def conv2D(img, kernel, pad=(0, 0), stride=1):
    H, W = img.shape
    kernel_h, kernel_w = kernel.shape
    out_h = (H + 2 * pad[0] - kernel_h) // stride + 1
    out_w = (W + 2 * pad[1] - kernel_w) // stride + 1
    new_img = np.pad(img, [[pad[0], pad[0]], [pad[1], pad[1]]], 'constant', constant_values=(0, 0))
    col = np.zeros((kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        for x in range(kernel_w):
            col[y, x, :, :] = new_img[y:(y + stride * out_h):stride, x:(x + stride * out_w):stride]

    new_img = col.transpose((2, 3, 0, 1)).reshape(out_h * out_w, -1)
    kernel = kernel.reshape((1, -1)).T
    return np.dot(new_img, kernel).T.reshape((out_h, out_w))


# 非极大值抑制
def NMS(Gx, Gy):
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    Gx[np.abs(Gx) <= 1e-5] = 1e-5
    temp = np.arctan(Gy, Gx) / np.pi * 180.
    temp[temp < -22.5] += 180.
    angle = np.zeros_like(temp, dtype=np.uint8)
    angle[np.where(temp <= 22.5)] = 0
    angle[np.where((temp > 22.5) & (temp <= 67.5))] = 45
    angle[np.where((temp > 67.5) & (temp <= 112.5))] = 90
    angle[np.where((temp > 112.5) & (temp <= 157.5))] = 135

    H, W = angle.shape
    di0, dj0, di1, dj1 = 0, 0, 0, 0
    for i in range(H):
        for j in range(W):
            if angle[i, j] == 0:
                di0, di1, dj0, dj1 = -1, 0, 1, 0
            elif angle[i, j] == 45:
                di0, di1, dj0, dj1 = -1, 1, 1, -1
            elif angle[i, j] == 90:
                di0, di1, dj0, dj1 = 0, -1, 0, 1
            elif angle[i, j] == 135:
                di0, di1, dj0, dj1 = -1, -1, 1, 1

            if j == 0:
                di0 = max(di0, 0)
                dj0 = max(dj0, 0)
            if j == W - 1:
                di0 = min(di0, 0)
                dj0 = min(dj0, 0)
            if i == 0:
                di1 = max(di1, 0)
                dj1 = max(dj1, 0)
            if i == H - 1:
                di1 = min(di1, 0)
                dj1 = min(dj1, 0)

            if max(max(G[i, j], G[i + di1, j + di0]), G[i + dj1, j + dj0]) != G[i, j]:
                G[i, j] = 0
    return G

# 直方图阈值
def histBasedThreshold(img):
    img = np.uint8(img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    peaks = np.where((hist[:-2] > hist[1:-1]) & (hist[1:-1] < hist[2:]))[0] + 1
    if len(peaks) >= 2:
        threshold = (peaks[0] + peaks[1]) // 2
    else:
        threshold = np.argmax(hist)
    _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh_img


# 双阈值检测
def doubleThresholdDetection(src, HT, LT):
    H, W = src.shape
    src[src >= HT] = 255
    src[src < LT] = 0
    src = np.pad(src,[[1, 1], [1, 1]], 'constant', constant_values=(0, 0))
    n8 = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)
    for i in range(1, H + 2):
        for j in range(1, W + 2):
            if LT <= src[i,j] <= HT:
                if np.max(src[i-1:i+2, j-1:j+2] * n8) >= HT:
                    src[i, j] = 255
                else:
                    src[i, j] = 0

    return src[1:H+1, 1:W+1]


# Sobel算子
def soberGrad(img):
    sober_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sober_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    result_x = conv2D(img, sober_x)
    result_y = conv2D(img, sober_y)
    return result_x, result_y


def main():
    # 参数设置
    i = [1, 2, 3]
    k = 1 # 高斯滤波器的大小
    sigma = 1.4 # 高斯滤波器的标准差

    # Hough算法的参数
    param1 = 100
    param2 = 40
    minRadius = 50
    maxRadius = 200

    # Canny算法，检测硬币的边缘
    for it in i:
        print(f">>>>>> 开始处理第", it, "张图片 >>>>>>")
        img_path = f"./input/input_{it}.jpg"
        image = Image.open(img_path)
        raw = np.copy(image)
        img = raw.mean(axis=2)

        kernel = gaussianKernel(k, sigma)
        img = conv2D(img, kernel, pad=(1, 1))
        Gx, Gy = soberGrad(img)
        G = NMS(Gx, Gy)
        G = doubleThresholdDetection(src=G, HT=100, LT=10)

        # Hough算法，寻找硬币的圆心坐标和半径
        hough = cv2.HoughCircles(np.uint8(G), cv2.HOUGH_GRADIENT, 1, 400, param1, param2, minRadius, maxRadius)
        print(f"共检测到{len(hough[0])}个硬币")
        cnt = 1
        for circle in hough[0]:
            x, y, r = circle
            cv2.circle(raw, (int(x), int(y)), int(r), (0, 0, 255), 3)
            cv2.circle(raw, (int(x), int(y)), 5, (0, 255, 0), -1)
            print(f"第{cnt}枚硬币：", "圆心 = ", (int(x), int(y)), "\t, 半径 = ", int(r))

        output_folder = f"./output/doubleThresholdDetection/{k}-{sigma}__100-10__{param1}-{param2}-{minRadius}-{maxRadius}/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cv2.imwrite(f"{output_folder}output_{it}.jpg", raw)
        print("")