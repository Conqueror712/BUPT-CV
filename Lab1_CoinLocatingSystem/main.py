import numpy as np
import cv2

# 自行编写的Canny边缘检测模块
def canny_edge_detection(image, sigma=1, low_threshold=50, high_threshold=150):
    # 使用高斯滤波平滑图像
    blurred_image = gaussian_blur(image, sigma)

    # 计算图像梯度
    gradient_magnitude, gradient_direction = calculate_gradient(blurred_image)

    # 非极大值抑制
    nms_image = non_max_suppression(gradient_magnitude, gradient_direction)

    # 双阈值处理
    thresholded_image = double_threshold(nms_image, low_threshold, high_threshold)

    # 边缘跟踪
    edge_image = edge_tracking(thresholded_image)

    return edge_image

def gaussian_blur(image, sigma):
    # 使用高斯核进行图像滤波
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)
    return blurred_image

def calculate_gradient(image):
    # 使用Sobel算子计算图像梯度
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    # 非极大值抑制
    rows, cols = gradient_magnitude.shape
    nms_image = np.zeros((rows, cols), dtype=np.uint8)
    angle = gradient_direction * (180.0 / np.pi)
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            # 根据梯度方向决定相邻像素
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            # 如果当前像素是局部最大值，保留，否则置为0
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                nms_image[i, j] = gradient_magnitude[i, j]
            else:
                nms_image[i, j] = 0

    return nms_image

def double_threshold(image, low_threshold, high_threshold):
    # 双阈值处理
    high_threshold = image.max() * high_threshold
    low_threshold = high_threshold * low_threshold

    rows, cols = image.shape
    result_image = np.zeros((rows, cols), dtype=np.uint8)

    weak = np.uint8(50)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)

    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result_image[strong_i, strong_j] = strong
    result_image[weak_i, weak_j] = weak

    return result_image

def edge_tracking(image):
    # 边缘跟踪
    rows, cols = image.shape
    result_image = np.zeros((rows, cols), dtype=np.uint8)

    weak = 50
    strong = 255

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == weak:
                if (image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong) \
                    or (image[i, j-1] == strong) or (image[i, j+1] == strong) or (image[i-1, j-1] == strong) \
                    or (image[i-1, j] == strong) or (image[i-1, j+1] == strong):
                    result_image[i, j] = strong
                else:
                    result_image[i, j] = 0
            elif image[i, j] == strong:
                result_image[i, j] = strong

    return result_image

# 自行编写的Hough圆检测模块
def hough_circle_detection(image, min_radius, max_radius):
    # 获取图像尺寸
    height, width = image.shape

    # 计算圆的最小距离
    min_dist = max(min(height, width) // 2, 20)  # 将minDist设置为图像长宽的一半或20（取大者）

    # 使用霍夫圆变换检测图像中的圆形
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]  # 返回圆的参数（圆心坐标和半径）
    else:
        return None  # 如果未检测到圆，返回None

# 主函数
def main():
    # 读取输入图像
    input_image = cv2.imread('./images/test_02.jpg', cv2.IMREAD_GRAYSCALE)

    # 边缘检测
    edge_image = canny_edge_detection(input_image)

    # 圆形检测
    circles = hough_circle_detection(edge_image, min_radius=10, max_radius=50)

    if circles is not None:  # 检查是否检测到圆
        # 打印圆心坐标和半径
        for circle in circles:
            print("Circle center:", (circle[0], circle[1]))
            print("Radius:", circle[2])
    else:
        print("No circles detected.")

if __name__ == "__main__":
    main()
