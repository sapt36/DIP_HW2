import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QScrollArea,
                             QSizePolicy, QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('圖像處理程式')
        self.image = None
        self.initUI()
        self.setGeometry(100, 100, 1000, 800)  # 設定較大視窗尺寸

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)  # 置中所有 UI 元素

        # 創建滾動視窗區域
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # 顯示圖片的 QLabel
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # 置中顯示圖片
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(False)

        # 將 QLabel 添加到滾動區域中
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        # 載入圖片按鈕
        load_button = QPushButton('載入圖片', self)
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        # 灰階處理按鈕
        gray_button = QPushButton('轉換為灰階圖片', self)
        gray_button.clicked.connect(self.convert_to_grayscale)
        layout.addWidget(gray_button)

        # 直方圖顯示按鈕
        hist_button = QPushButton('顯示灰階直方圖', self)
        hist_button.clicked.connect(self.show_histogram)
        layout.addWidget(hist_button)

        # 手動二值化按鈕
        thresh_button = QPushButton('手動二值化圖片', self)
        thresh_button.clicked.connect(self.manual_threshold)
        layout.addWidget(thresh_button)

        # 調整亮度與對比按鈕
        adjust_button = QPushButton('調整亮度與對比', self)
        adjust_button.clicked.connect(self.open_brightness_contrast_dialog)
        layout.addWidget(adjust_button)

        self.setLayout(layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '選擇圖片', '', 'Image files (*.jpg *.jpeg *.bmp)')
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is None:
                self.image_label.setText("無法載入圖片")
            else:
                self.display_image(self.image)

    def display_image(self, img):
        """ 將 OpenCV 圖片轉換為 QImage 並顯示，保持原始大小 """
        qformat = QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8
        h, w = img.shape[:2]
        img = QImage(img.data, w, h, img.strides[0], qformat)
        img = img.rgbSwapped()  # OpenCV 使用 BGR，因此我們要轉換成 RGB

        # 將圖片轉換為 QPixmap 並顯示，保持圖片原尺寸
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()  # 調整 QLabel 大小以適應圖片原始尺寸

    def convert_to_grayscale(self):
        if self.image is not None:
            gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(gray_img)

    def show_histogram(self):
        if self.image is not None:
            gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.title('Gray Scale Histogram')
            plt.show()

    def manual_threshold(self):
        if self.image is not None:
            gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
            self.display_image(binary_img)

    def open_brightness_contrast_dialog(self):
        if self.image is not None:
            dialog = BrightnessContrastDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                brightness, contrast = dialog.get_values()
                self.adjust_brightness_contrast(brightness, contrast)

    def adjust_brightness_contrast(self, brightness, contrast):
        if self.image is not None:
            adjusted_img = cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)
            self.display_image(adjusted_img)


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


def main():
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
