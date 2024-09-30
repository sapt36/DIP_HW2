import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QScrollArea,
                             QSizePolicy, QDialog, QHBoxLayout, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt


class ImageProcessingApp(QWidget):

    # 主視窗UI啟動
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HW2 圖像處理軟體')
        self.image = None
        self.initUI()
        self.setGeometry(100, 100, 1500, 900)  # 設置視窗尺寸

    # 主視窗UI設置
    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)  # 置中所有 UI 元素

        # 設置字體
        self.setFont(QFont('微軟正黑體', 12, QFont.Bold))

        # 創建滾動視窗區域
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # 創建一個容器 widget 放入滾動區域
        container_widget = QWidget(self.scroll_area)
        self.scroll_area.setWidget(container_widget)

        # 創建佈局來居中顯示圖片
        container_layout = QVBoxLayout(container_widget)
        container_layout.setAlignment(Qt.AlignCenter)

        # 顯示圖片的 QLabel
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # 置中顯示圖片
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(False)

        # 將 QLabel 添加到容器佈局中
        container_layout.addWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        # 載入圖片按鈕
        load_button = QPushButton('1. 載入並顯示彩色 BMP 或 JPEG圖片', self)
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        # 灰階處理按鈕
        gray_button = QPushButton('2. 使用兩種公式將圖片轉換為灰階 並使用圖像相減做比較結果顯示', self)
        gray_button.clicked.connect(self.convert_to_grayscale_and_compare)
        layout.addWidget(gray_button)

        # 直方圖顯示按鈕
        hist_button = QPushButton('3. 顯示灰階直方圖', self)
        hist_button.clicked.connect(self.show_histogram)
        layout.addWidget(hist_button)

        # 手動二值化按鈕
        thresh_button = QPushButton('4. 將灰階圖像轉換為二值化圖像', self)
        thresh_button.clicked.connect(self.manual_threshold)
        layout.addWidget(thresh_button)

        # 調整空間解析度按鈕
        resize_button = QPushButton('5-1. 調整空間解析度（放大或縮小）', self)
        resize_button.clicked.connect(self.open_resolution_dialog)
        layout.addWidget(resize_button)

        # 調整灰階級別按鈕
        grayscale_levels_button = QPushButton('5-2. 調整灰階級別 (2 的 1 bit - 8 bits)', self)
        grayscale_levels_button.clicked.connect(self.open_grayscale_dialog)
        layout.addWidget(grayscale_levels_button)

        # 調整亮度與對比度按鈕
        adjust_button = QPushButton('6. 調整亮度與對比度', self)
        adjust_button.clicked.connect(self.open_brightness_contrast_dialog)
        layout.addWidget(adjust_button)

        # 直方圖均衡化按鈕
        hist_equalize_button = QPushButton('7. 自動對比度調整(直方圖均衡化)', self)
        hist_equalize_button.clicked.connect(self.histogram_equalization_color)
        layout.addWidget(hist_equalize_button)

        self.setLayout(layout)

    # 載入圖片，並顯示在主視窗上
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '選擇圖片', '', 'Image files (*.jpg *.jpeg *.bmp)')
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is None:
                self.image_label.setText("無法載入圖片")
            else:
                self.display_image(self.image)

    # 將 OpenCV 圖片轉換為 QImage 並顯示，保持原始大小
    def display_image(self, img):
        qformat = QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8
        h, w = img.shape[:2]
        img = QImage(img.data, w, h, img.strides[0], qformat)
        img = img.rgbSwapped()  # OpenCV 使用 BGR，因此我們要轉換成 RGB

        # 將圖片轉換為 QPixmap 並顯示，保持圖片原尺寸
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())  # 調整 QLabel 大小以適應圖片原始尺寸

    # 轉換圖片為灰階並比較使用兩種公式生成的灰階圖像
    def convert_to_grayscale_and_compare(self):
        if self.image is not None:
            # 獲取圖像的高、寬以及 RGB 通道數
            h, w, _ = self.image.shape
            self.image = self.image.astype(np.uint16)  # 使用 uint16 防止 overflow
            # 公式 A: GRAY = (R + G + B) / 3.0
            gray_avg = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    B, G, R = self.image[i, j]
                    gray_avg[i, j] = (R + G + B) / 3

            # 公式 B: GRAY = 0.299*R + 0.587*G + 0.114*B
            gray_weighted = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    B, G, R = self.image[i, j]
                    gray_weighted[i, j] = 0.299 * R + 0.587 * G + 0.114 * B

            # 顯示兩張灰階圖片到彈出視窗
            dialog = QDialog(self)
            dialog.setWindowTitle('灰階轉換結果')
            dialog.setGeometry(100, 100, 1000, 600)

            layout = QHBoxLayout()

            # 將公式 A 的灰階圖像顯示
            gray_avg_label = QLabel(dialog)
            pixmap_avg = self.convert_cv_to_pixmap(gray_avg)
            gray_avg_label.setPixmap(pixmap_avg)
            gray_avg_label.adjustSize()  # 確保 QLabel 大小適應圖片
            layout.addWidget(gray_avg_label)

            # 將公式 B 的灰階圖像顯示
            gray_weighted_label = QLabel(dialog)
            pixmap_weighted = self.convert_cv_to_pixmap(gray_weighted)
            gray_weighted_label.setPixmap(pixmap_weighted)
            gray_weighted_label.adjustSize()
            layout.addWidget(gray_weighted_label)

            dialog.setLayout(layout)
            dialog.exec_()

            # 比較兩張灰階圖像的差異，並顯示到主視窗
            difference = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    diff_value = abs(int(gray_avg[i, j]) - int(gray_weighted[i, j]))
                    difference[i, j] = np.clip(diff_value, 0, 255)  # 確保在 0-255 範圍內
            self.display_image(difference)

    # 將 OpenCV 圖像轉換為 QPixmap
    def convert_cv_to_pixmap(self, cv_img):
        qformat = QImage.Format_RGB888 if len(cv_img.shape) == 3 else QImage.Format_Grayscale8
        h, w = cv_img.shape[:2]
        img = QImage(cv_img.data, w, h, cv_img.strides[0], qformat)
        if len(cv_img.shape) == 3:  # 如果是彩色圖像，進行 RGB 轉換
            img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        return pixmap

    # 彈出調整亮度與對比的對話框
    def open_brightness_contrast_dialog(self):
        if self.image is not None:
            dialog = BrightnessContrastDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                brightness, contrast = dialog.get_values()
                self.adjust_brightness_contrast(brightness, contrast)

    # 調整圖片的亮度與對比度
    def adjust_brightness_contrast(self, brightness, contrast):
        if self.image is not None:
            # 使用 np.clip() 限制 np 陣列內元素大小範圍在 0-255 之間
            adjusted_img = np.clip(self.image * contrast + brightness, 0, 255).astype(np.uint8)
            self.display_image(adjusted_img)

    # 將灰階圖像轉換為二值化圖像
    def manual_threshold(self):
        if self.image is not None:
            # 創建並顯示閾值彈出視窗
            dialog = ThresholdDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                threshold_value = dialog.get_value()
                if threshold_value is not None:
                    # 手動將圖片轉為灰階
                    gray_img = self.image.dot([0.299, 0.587, 0.114])

                    # 建立與灰階圖像相同大小的空陣列來存放二值化結果
                    binary_img = [[0 for _ in range(len(gray_img[0]))] for _ in range(len(gray_img))]

                    # 遍歷灰階圖像中的每個像素
                    for i in range(len(gray_img)):
                        for j in range(len(gray_img[0])):
                            # 判斷像素值是否大於閾值
                            if gray_img[i][j] > threshold_value:
                                binary_img[i][j] = 255  # 設置為白色
                            else:
                                binary_img[i][j] = 0  # 設置為黑色

                    # 將二值化結果轉換為 numpy 陣列，並設置資料型態為 uint8
                    binary_img = np.array(binary_img, dtype=np.uint8)

                    # 顯示二值化後的圖像
                    self.display_image(binary_img)

    # 計算並顯示灰階圖像對應之直方圖
    def show_histogram(self):
        if self.image is not None:
            # Convert the image to grayscale
            gray_img = self.image.dot([0.299, 0.587, 0.114])
            # Calculate histogram
            hist, bins = np.histogram(gray_img.flatten(), bins=256, range=[0, 256])
            # Plot histogram as a bar plot
            plt.bar(bins[:-1], hist, width=1, edgecolor='black')
            plt.title('Gray Scale Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.show()

    # 彈出輸入放大或縮小倍數的對話框
    def open_resolution_dialog(self):
        dialog = ResolutionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            scale_factor = dialog.get_value()
            self.adjust_resolution(scale_factor)

    # 調整空間解析度函數
    def adjust_resolution(self, scale_factor):
        if self.image is not None:
            # 取得圖像的原始尺寸
            h, w = self.image.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)

            resized_image = self.nearest_neighbor_interpolation(new_h, new_w, scale_factor)

            # 顯示調整後的圖像
            self.display_image(resized_image)

    # 使用最近鄰插值實現放大功能
    def nearest_neighbor_interpolation(self, new_h, new_w, scale_factor):
        # 生成新的網格坐標，減少逐像素迴圈，利用向量化操作
        x_indices = (np.arange(new_h) / scale_factor).astype(int)
        y_indices = (np.arange(new_w) / scale_factor).astype(int)

        # 限制索引範圍，防止溢出，減少邊界檢查
        x_indices = np.clip(x_indices, 0, self.image.shape[0] - 1)
        y_indices = np.clip(y_indices, 0, self.image.shape[1] - 1)

        # 使用 NumPy 批量索引和切片
        resized_image = self.image[x_indices[:, None], y_indices]

        # 確保結果圖像的資料類型為 uint8，並裁剪範圍
        resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)

        return resized_image

    # 彈出輸入灰階級別的對話框
    def open_grayscale_dialog(self):
        dialog = GrayscaleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            levels = dialog.get_value()
            self.adjust_grayscale_levels(levels)

    # 調整灰階級別函數
    def adjust_grayscale_levels(self, levels):
        if self.image is not None:
            # 將圖片轉為灰階
            gray_img = self.image.dot([0.299, 0.587, 0.114]).astype(np.uint8)

            # 重新分佈灰階級別
            gray_img = np.floor_divide(gray_img, 256 // levels) * (256 // levels)

            # 顯示調整後的圖片
            self.display_image(gray_img)

    # 直方圖均衡化函數（對三個通道均衡化）
    def histogram_equalization_color(self):
        if self.image is not None:
            # Create an empty result image
            equalized_img = np.zeros_like(self.image)

            # Perform equalization for R, G, B channels
            for channel in range(3):
                channel_data = self.image[:, :, channel]
                # Calculate histogram
                hist, bins = np.histogram(channel_data.flatten(), 256, [0, 256])

                # Calculate cumulative distribution function (CDF)
                cdf = hist.cumsum()
                cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to [0, 255]

                # Map pixel values using the CDF
                equalized_channel = np.zeros_like(channel_data)
                for intensity in range(256):
                    equalized_channel[channel_data == intensity] = cdf_normalized[intensity]

                # Store the equalized channel in the result image
                equalized_img[:, :, channel] = equalized_channel

            # Display the equalized image
            self.display_image(equalized_img.astype(np.uint8))

            # Display the histogram of the equalized grayscale image
            gray_img = equalized_img.dot([0.299, 0.587, 0.114]).astype(np.uint8)  # Convert image to grayscale
            hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])  # Calculate grayscale histogram
            plt.figure()
            plt.bar(bins[:-1], hist, width=1, edgecolor='black', color='gray')
            plt.title('Histogram of Grayscale Values after Equalization')
            plt.xlabel('Gray Level')
            plt.ylabel('Frequency')
            plt.show()


# 創建一個對話框(彈出視窗)，使用者可以輸入亮度和對比度的數值調整圖片，並點擊確認或取消按鈕。
class BrightnessContrastDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('調整亮度與對比')
        self.brightness_input = QLineEdit(self)
        self.contrast_input = QLineEdit(self)

        # 設置範圍提示
        self.brightness_input.setPlaceholderText('亮度: -100 至 100')
        self.contrast_input.setPlaceholderText('對比: 0.0 至 3.0')

        form_layout = QFormLayout()
        form_layout.addRow('亮度:', self.brightness_input)
        form_layout.addRow('對比:', self.contrast_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_values(self):
        try:
            brightness = int(self.brightness_input.text())
            contrast = float(self.contrast_input.text())

            # 檢查範圍
            if not (-100 <= brightness <= 100):
                raise ValueError("亮度必須在 -100 至 100 之間")
            if not (0.0 <= contrast <= 3.0):
                raise ValueError("對比必須在 0.0 至 3.0 之間")
            return brightness, contrast
        except ValueError as e:
            QMessageBox.warning(self, '輸入錯誤', str(e))
            return None, None


# 創建一個對話框，可輸入參數調整放大或縮小倍數 (0.1 - 5.0)
class ResolutionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('調整圖片大小')
        self.scale_input = QLineEdit(self)
        self.scale_input.setPlaceholderText('倍數 (0.1 - 5.0)')

        form_layout = QFormLayout()
        form_layout.addRow('圖片大小倍數:', self.scale_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_value(self):
        try:
            scale_factor = float(self.scale_input.text())
            if not (0.1 <= scale_factor <= 5.0):
                raise ValueError("倍數必須在 0.1 到 5.0 之間")
            return scale_factor
        except ValueError as e:
            QMessageBox.warning(self, '輸入錯誤', str(e))
            return None


# 創建一個對話框，可輸入參數調整灰階級別(2 - 256)
class GrayscaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('調整灰階級別')
        self.levels_input = QLineEdit(self)
        self.levels_input.setPlaceholderText('灰階級別 (2 - 256)')

        form_layout = QFormLayout()
        form_layout.addRow('灰階級別:', self.levels_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_value(self):
        try:
            levels = int(self.levels_input.text())
            if not (2 <= levels <= 256):
                raise ValueError("灰階級別必須在 2 到 256 之間")
            return levels
        except ValueError as e:
            QMessageBox.warning(self, '輸入錯誤', str(e))
            return None


# 創建一個對話框，可輸入參數調整二值化閥值(2 - 256)
class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('輸入閾值')
        self.threshold_input = QLineEdit(self)
        self.threshold_input.setPlaceholderText('閾值 (0 - 255)')

        # 建立表單佈局
        form_layout = QFormLayout()
        form_layout.addRow('二值化閾值:', self.threshold_input)

        # 建立按鈕
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # 組合佈局
        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_value(self):
        try:
            threshold_value = int(self.threshold_input.text())
            if not (0 <= threshold_value <= 255):
                raise ValueError("閾值必須在 0 到 255 之間")
            return threshold_value
        except ValueError as e:
            QMessageBox.warning(self, '輸入錯誤', str(e))
            return None


# Main Execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
