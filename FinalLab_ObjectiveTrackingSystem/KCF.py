import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt


class CustomHOG:
    def __init__(self, window_size=(64, 128), block_size=(8, 8), block_stride=(4, 4), cell_size=(4, 4), bins=9):
        """
        初始化高级HOG描述符。
        :param window_size: 检测窗口大小
        :param block_size: 块大小
        :param block_stride: 块步长
        :param cell_size: 单元格（cell）大小
        :param bins: 方向直方图的数量
        """
        self.window_size = window_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nBins = bins
        # 创建HOG描述符对象
        self.hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, bins)

    def extract_feature(self, image):
        """
        提取图像的HOG特征。
        :param image: 输入图像
        :return: HOG特征描述符
        """
        hog_features = self.hog.compute(image, winStride=self.block_stride, padding=(0, 0))  # 设置窗口步长（winStride）为窗口大小

        # 保证块完整地位于窗口内部，并不会超出边界
        h = self.window_size[1] // self.block_stride[1] - 1
        w = self.window_size[0] // self.block_stride[0] - 1

        return hog_features.reshape(w, h, 4*self.nBins).transpose(2, 1, 0)  # 每个块由4个cell组成，每个cell有9个bin


def visualize_features(image_path):
    """
    可视化HOG特征。
    :param image_path: 输入图像
    """
    image = cv2.imread(image_path)
    hog_descriptor = CustomHOG()
    features = hog_descriptor.extract_feature(image)
    print("Feature vector length:", len(features))
    h, w = image.shape[:2]
    radius = min(h, w) // 2
    center = (w // 2, h // 2)
    output_image = np.zeros_like(image)
    for i in range(len(features)):
        angle = 2 * np.pi * i / len(features)
        x = int(center[0] + radius * np.cos(angle) * features[i])
        y = int(center[1] + radius * np.sin(angle) * features[i])
        cv2.line(output_image, center, (x, y), (255, 255, 255), 2)
    cv2.imshow("HOG Features", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_color_features(image):
    """
    提取颜色直方图特征。
    :param image: 输入图像
    :return: 颜色直方图特征
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def visualize_color_histogram(image_path):
    """
    可视化输入图像的颜色直方图特征。
    :param image_path: 输入图像路径
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colors = ('Hue', 'Saturation', 'Value')
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([hsv_image], [i], None, [256], [0, 256])
        plt.plot(hist, label=color)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title("Color Histogram Features")
    plt.show()

def extract_edge_features(image):
    """
    提取边缘特征。
    :param image: 输入图像
    :return: 边缘特征描述子
    """
    edges = cv2.Canny(image, 100, 200)
    return edges

def visualize_edges(image_path):
    """
    可视化输入图像的边缘特征。
    :param image_path: 输入图像路径
    """
    image = cv2.imread(image_path)
    edges = extract_edge_features(image)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Features")
    plt.show()

# 可视化特征
# visualize_features(image_path = "../data/football/00000001.jpg")
# visualize_color_histogram(image_path = "../data/football/00000001.jpg")
# visualize_edges(image_path = "../data/football/00000001.jpg")

class KCFTracker:
    def __init__(self, sigma=0.5, lambdar=0.0001, update_rate=0.012, scale_factor=2.5):
        # 超参数设置
        self.max_patch_size = 256
        self.scale_factor = scale_factor  # 扩展搜索窗口的比例因子
        self.sigma = sigma  # 高斯核的带宽
        self.lambdar = lambdar  # 正则化项
        self.update_rate = update_rate  # 更新速率
        self.sigma_factor = 0.125  # 高斯的标准偏差因子

        # KCF算法变量定义
        self.scale_h = 0.  # 高度的缩放因子
        self.scale_w = 0.  # 宽度的缩放因子
        self.ph = 0  # 特征的高度
        self.pw = 0  # 特征的宽度
        self.hog = CustomHOG((self.pw, self.pw))
        self.alphaf = None  # 模型的频域表示
        self.x = None  # 当前特征
        self.roi = None  # 当前目标的位置

    def init_param(self, image, roi):
        """
        对视频的第一帧进行标记，更新tracer的参数
        :param image: 第一帧图像
        :param roi: 初始区域（x1, y1, w, h）
        :return: None
        """
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        # 确定Patch的大小，并在此Patch中提取HOG特征描述子
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = CustomHOG((self.pw, self.ph))

        # 提取特征和目标的高斯响应
        x = self.get_feature(image, roi)
        y = self.gaussian_matrix(x.shape[2], x.shape[1])

        # 根据得到的核相关和高斯响应计算新的滤波器系数alphaf
        self.alphaf = fft2(y) / (fft2(self.compute_kernel_correlation(x, x, self.sigma)) + self.lambdar)
        self.x = x
        self.roi = roi

    def update_frame(self, image):
        """
        更新跟踪器状态。
        :param image: 新的帧图像
        """
        # 包含矩形框信息的四元组(x, y, w, h)
        center_x, center_y, w, h = self.roi
        max_response = -1   # 初始化最大响应值
        # best_scale = 1.0
        #
        # # 定义尺度金字塔
        # scales = [0.83, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        #
        # # 遍历所有尺度
        # for scale in scales:
        #     scaled_w = int(w * scale)
        #     scaled_h = int(h * scale)
        #     scaled_x = int(cur_x + w / 2 - scaled_w / 2)
        #     scaled_y = int(cur_y + h / 2 - scaled_h / 2)
        #
        #     # 提取缩放级别的特征
        #     z = self.get_feature(image, map(int,(scaled_x, scaled_y, scaled_w, scaled_h)))
        #     # 计算响应
        #     responses = np.real(ifft2(self.alphaf * fft2(self.kernel_correlation(self.x, z, self.sigma))))
        #     # 找到最大响应
        #     res = np.max(responses)
        #     if res > max_response:
        #         max_response = res
        #         best_scale = scale
        #         best_roi = (scaled_x, scaled_y, scaled_w, scaled_h)
        #         best_z = z
        #
        # # 更新ROI和特征
        # self.roi = best_roi
        # self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        # y = self.gaussian_matrix(best_z.shape[2], best_z.shape[1])
        # new_alphaf = fft2(y) / (fft2(self.kernel_correlation(best_z, best_z, self.sigma)) + self.lambdar)
        # self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate
        #
        # return self.roi

        for scale in [0.95, 1.0, 1.05]:
            # 将ROI值处理为整数
            roi = map(int, (center_x, center_y, w * scale, h * scale))

            z = self.get_feature(image, roi)    # tuple(4*nBins, h, w)
            # 计算响应
            responses = np.real(ifft2(self.alphaf * fft2(self.compute_kernel_correlation(self.x, z, self.sigma))))  # 傅里叶逆变换的实部，它代表了位置的响应强度
            height, width = responses.shape
            idx = np.argmax(responses)

            # 评估响应值
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z

        # 更新矩形框的相关参数
        self.roi = (center_x + dx, center_y + dy, best_w, best_h)

        # 更新参数alphaf和x
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_matrix(best_z.shape[2], best_z.shape[1])
        new_alphaf = fft2(y) / (fft2(self.compute_kernel_correlation(best_z, best_z, self.sigma)) + self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        """
        对特征进行采样
        :param image:
        :param roi: 包含矩形框信息的四元组(x, y, w, h)
        :return:
        """
        # 扩大ROI区域
        center_x, center_y, width, height = roi
        width = int(width * self.scale_factor) // 2 * 2
        height = int(height * self.scale_factor) // 2 * 2
        top_left_x = int(center_x - width // 2)
        top_left_y = int(center_y - height // 2)

        # 截取并调整图像大小
        cropped_img = image[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
        resized_img = cv2.resize(cropped_img, (self.pw, self.ph))

        # 提取HOG特征
        feature = self.hog.extract_feature(resized_img)

        # 获取特征的尺寸
        channels, feature_height, feature_width = feature.shape
        self.scale_h = float(feature_height) / height
        self.scale_w = float(feature_width) / width

        # 应用Hann窗函数进行加权
        hann_window = np.outer(np.hanning(feature_height), np.hanning(feature_width))
        hann_window_3d = np.repeat(hann_window[np.newaxis, :, :], channels, axis=0)  # 创建一个与通道数相同的堆叠窗

        weighted_feature = feature * hann_window_3d  # 确保维度匹配

        return weighted_feature

    def gaussian_matrix(self, width, height):
        """
        :param width:
        :param height:
        :return:      一个w*h的高斯矩阵
        """
        effective_sigma = np.sqrt(width * height) / self.scale_factor * self.sigma_factor
        center_y, center_x = height // 2, width // 2

        # 使用meshgrid生成坐标网格，并转换为浮点型以支持后续计算
        grid_y, grid_x = np.mgrid[-center_y:-center_y + height, -center_x:-center_x + width].astype(np.float32)
        grid_x += (1 - width % 2) / 2.0
        grid_y += (1 - height % 2) / 2.0

        # 计算高斯函数
        norm_factor = 1.0 / (2.0 * np.pi * effective_sigma ** 2)
        gaussian_matrix = norm_factor * np.exp(-((grid_x ** 2 + grid_y ** 2) / (2.0 * effective_sigma ** 2)))

        return gaussian_matrix

    def compute_kernel_correlation(self, features1, features2, bandwidth, kernel = 'radial_basis'):
        """
        计算两组特征之间的核相关。
        :param features1: 第一组特征
        :param features2: 第二组特征
        :param bandwidth: 高斯带宽参数
        :return: 计算得到的核相关矩阵
        """
        if kernel == 'radial_basis':
            # 将特征转换到频域
            fft_features1 = fft2(features1)
            fft_features2 = fft2(features2)

            # 计算两个特征的互相关
            cross_correlation = np.conj(fft_features1) * fft_features2

            # 通过逆傅里叶变换将结果转换回实空间
            ifft_result = ifft2(np.sum(cross_correlation, axis=0))

            # 移动频谱中心到数组的中心
            ifft_centered = fftshift(ifft_result)

            # 计算径向基函数
            squared_difference = np.sum(features1 ** 2) + np.sum(features2 ** 2) - 2.0 * ifft_centered
            radial_basis_output = np.exp(-1 / bandwidth ** 2 * np.abs(squared_difference) / squared_difference.size)

            return radial_basis_output
        elif kernel == 'dot':
            return np.dot(features1.flatten(), features2.flatten())

        elif kernel == 'Polynomial':
            degree = 3  # 多项式核的阶数
            coef = 1  # 多项式核的常数项
            dot_product = np.dot(features1.flatten(), features2.flatten())
            return (dot_product + coef) ** degree

        else:
            raise ValueError("Invalid kernel type. Please choose from 'radial_basis', 'dot' and 'Polynomial'.")


# 应用示例
def main():
    # cap = cv2.VideoCapture("../data/car.avi")
    cap = cv2.VideoCapture("car.mp4")
    _, frame = cap.read()
    # roi = cv2.selectROI(frame)
    roi = cv2.selectROI("tracking", frame, False, False)

    tracker = KCFTracker()
    tracker.init_param(frame, roi)

    while True:
        _, frame = cap.read()
        x, y, w, h = tracker.update_frame(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
