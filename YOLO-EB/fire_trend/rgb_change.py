import os
import cv2
import numpy as np

def process_images(input_folder, output_red_folder, output_gray_folder):
    # 创建输出文件夹
    os.makedirs(output_red_folder, exist_ok=True)
    os.makedirs(output_gray_folder, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            input_path = os.path.join(input_folder, filename)
            red_output_path = os.path.join(output_red_folder, filename)
            gray_output_path = os.path.join(output_gray_folder, filename)

            # 读取图像
            img = cv2.imread(input_path)
            if img is None:
                print(f"无法读取图像: {filename}")
                continue

            # 提取红色通道
            red_channel = img[:, :, 2]

            # 创建彩色红色通道图（其他通道设为0）
            red_only_img = np.zeros_like(img)
            red_only_img[:, :, 2] = red_channel  # 只保留红色

            # 保存红色图和彩色灰度图
            cv2.imwrite(red_output_path, red_only_img)
            cv2.imwrite(gray_output_path, red_channel)  # 灰度图像

            print(f"处理完成: {filename}")

# 示例目录
input_folder = 'RGB'                # 原始RGB图片文件夹
output_red_folder = 'red_channel'        # 彩色红通道图输出目录
output_gray_folder = 'red_channel_gray'       # 红通道灰度图输出目录

process_images(input_folder, output_red_folder, output_gray_folder)
