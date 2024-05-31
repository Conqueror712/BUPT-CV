import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from flask import Flask, render_template, Response
import time

app = Flask(__name__)

class CustomHOG:
    def __init__(self, window_size=(64, 128), block_size=(8, 8), block_stride=(4, 4), cell_size=(4, 4), bins=9):
        self.window_size = window_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nBins = bins
        self.hog = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, bins)

    def extract_feature(self, image):
        hog_features = self.hog.compute(image, winStride=self.block_stride, padding=(0, 0))
        h = self.window_size[1] // self.block_stride[1] - 1
        w = self.window_size[0] // self.block_stride[0] - 1
        return hog_features.reshape(w, h, 4*self.nBins).transpose(2, 1, 0)

class KCFTracker:
    def __init__(self, sigma=0.5, lambdar=0.0001, update_rate=0.012, scale_factor=2.5):
        self.max_patch_size = 256
        self.scale_factor = scale_factor
        self.sigma = sigma
        self.lambdar = lambdar
        self.update_rate = update_rate
        self.sigma_factor = 0.125
        self.scale_h = 0.
        self.scale_w = 0.
        self.ph = 0
        self.pw = 0
        self.hog = CustomHOG((self.pw, self.pw))
        self.alphaf = None
        self.x = None
        self.roi = None

    def init_param(self, image, roi):
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = CustomHOG((self.pw, self.ph))
        x = self.get_feature(image, roi)
        y = self.gaussian_matrix(x.shape[2], x.shape[1])
        self.alphaf = fft2(y) / (fft2(self.compute_kernel_correlation(x, x, self.sigma)) + self.lambdar)
        self.x = x
        self.roi = roi

    def update_frame(self, image):
        center_x, center_y, w, h = self.roi
        max_response = -1
        for scale in [0.95, 1.0, 1.05]:
            roi = map(int, (center_x, center_y, w * scale, h * scale))
            z = self.get_feature(image, roi)
            responses = np.real(ifft2(self.alphaf * fft2(self.compute_kernel_correlation(self.x, z, self.sigma))))
            height, width = responses.shape
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        self.roi = (center_x + dx, center_y + dy, best_w, best_h)
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_matrix(best_z.shape[2], best_z.shape[1])
        new_alphaf = fft2(y) / (fft2(self.compute_kernel_correlation(best_z, best_z, self.sigma)) + self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate
        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        center_x, center_y, width, height = roi
        width = int(width * self.scale_factor) // 2 * 2
        height = int(height * self.scale_factor) // 2 * 2
        top_left_x = int(center_x - width // 2)
        top_left_y = int(center_y - height // 2)
        cropped_img = image[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
        resized_img = cv2.resize(cropped_img, (self.pw, self.ph))
        feature = self.hog.extract_feature(resized_img)
        channels, feature_height, feature_width = feature.shape
        self.scale_h = float(feature_height) / height
        self.scale_w = float(feature_width) / width
        hann_window = np.outer(np.hanning(feature_height), np.hanning(feature_width))
        hann_window_3d = np.repeat(hann_window[np.newaxis, :, :], channels, axis=0)
        weighted_feature = feature * hann_window_3d
        return weighted_feature

    def gaussian_matrix(self, width, height):
        effective_sigma = np.sqrt(width * height) / self.scale_factor * self.sigma_factor
        center_y, center_x = height // 2, width // 2
        grid_y, grid_x = np.mgrid[-center_y:-center_y + height, -center_x:-center_x + width].astype(np.float32)
        grid_x += (1 - width % 2) / 2.0
        grid_y += (1 - height % 2) / 2.0
        norm_factor = 1.0 / (2.0 * np.pi * effective_sigma ** 2)
        gaussian_matrix = norm_factor * np.exp(-((grid_x ** 2 + grid_y ** 2) / (2.0 * effective_sigma ** 2)))
        return gaussian_matrix

    def compute_kernel_correlation(self, features1, features2, bandwidth, kernel = 'radial_basis'):
        if kernel == 'radial_basis':
            fft_features1 = fft2(features1)
            fft_features2 = fft2(features2)
            cross_correlation = np.conj(fft_features1) * fft_features2
            ifft_result = ifft2(np.sum(cross_correlation, axis=0))
            ifft_centered = fftshift(ifft_result)
            squared_difference = np.sum(features1 ** 2) + np.sum(features2 ** 2) - 2.0 * ifft_centered
            radial_basis_output = np.exp(-1 / bandwidth ** 2 * np.abs(squared_difference) / squared_difference.size)
            return radial_basis_output
        elif kernel == 'dot':
            return np.dot(features1.flatten(), features2.flatten())
        elif kernel == 'Polynomial':
            degree = 3
            coef = 1
            dot_product = np.dot(features1.flatten(), features2.flatten())
            return (dot_product + coef) ** degree
        else:
            raise ValueError("Invalid kernel type. Please choose from 'radial_basis', 'dot' and 'Polynomial'.")

def gen_frames():
    cap = cv2.VideoCapture("car.mp4")
    tracker = KCFTracker()
    ret, frame = cap.read()
    roi = cv2.selectROI("tracking", frame, False, False)
    tracker.init_param(frame, roi)
    
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, w, h = tracker.update_frame(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
