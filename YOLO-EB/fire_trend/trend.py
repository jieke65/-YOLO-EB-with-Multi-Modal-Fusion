import os  # 导入操作系统模块，用于处理文件和路径
import cv2  # 导入 OpenCV 库，用于图像处理
import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘图

# 定义最小二乘法拟合函数
def least_squares_fit(x, y):
    """使用最小二乘法进行线性拟合，返回斜率和截距"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

# 主函数：处理所有图片并判断火势趋势
def process_images_and_judge_trend(original_images_path, masks_path, output_txt_path, trend_txt_path, gray_plot_path, frame_rate=10):
    original_images = sorted(os.listdir(original_images_path))
    mask_images = sorted(os.listdir(masks_path))

    masks_dict = {}
    for mask_name in mask_images:
        if "_mask_" in mask_name:
            base_name = "_".join(mask_name.split("_")[:2])
            masks_dict.setdefault(base_name, []).append(os.path.join(masks_path, mask_name))

    red_channel_sums = []

    for original_img_name in original_images:
        original_img_path = os.path.join(original_images_path, original_img_name)
        base_name = os.path.splitext(original_img_name)[0]

        original_img = cv2.imread(original_img_path)
        if original_img is None:
            continue

        mask_paths = masks_dict.get(base_name, [])

        largest_gray_sum = 0

        for mask_path in mask_paths:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            mask_resized = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            red_channel = original_img[:, :, 2]
            masked_region = cv2.bitwise_and(red_channel, red_channel, mask=mask_resized)
            gray_sum = np.sum(masked_region)

            largest_gray_sum = max(largest_gray_sum, gray_sum)

        red_channel_sums.append(largest_gray_sum)

    frame_indices = np.arange(len(red_channel_sums))
    time_in_sec = frame_indices / frame_rate  # ✅ 时间单位：秒
    slopes = []
    fitted_values = []

    for i in range(0, len(frame_indices), 3):  # 每 3 帧进行一次拟合
        start = i
        end = min(i + 3, len(frame_indices))
        x_data = time_in_sec[start:end]
        y_data = red_channel_sums[start:end]

        if len(x_data) >= 2:
            slope, intercept = least_squares_fit(x_data, y_data)
            slopes.append(slope)
            fitted_values.extend(slope * x_data + intercept)
        else:
            slopes.append(0)
            fitted_values.extend(y_data)

    # 保存火势趋势分析结果
    with open(trend_txt_path, "w") as f:
        f.write("Fire Trend Analysis (Based on 3-frame segments, time in seconds):\n")
        for i, slope in enumerate(slopes):
            trend = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            f.write(f"Segment {i + 1}: Slope = {slope:.4f}, Trend = {trend}\n")

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(time_in_sec, red_channel_sums, label="Original Gray Value Sum", marker='o', color='r', linewidth=1.5)
    plt.plot(time_in_sec, fitted_values, label="Fitted Curve", linestyle='--', color='b', linewidth=1.5)
    plt.xlabel("Time (s)")  # ✅ 改为秒
    plt.ylabel("Gray Value Sum")
    plt.title("Red Channel Gray Value Sum with Fitted Curve (Fire Trend Analysis)")
    plt.grid(True)
    plt.legend()
    plt.savefig(gray_plot_path)
    plt.show()

    # 保存每帧灰度值总和
    with open(output_txt_path, "w") as f:
        f.write("Red Channel Gray Value Sum per Frame:\n")
        f.write("\n".join([f"Frame {i + 1} ({time_in_sec[i]:.2f} s): {gray_sum}" for i, gray_sum in enumerate(red_channel_sums)]))

    print(f"灰度值总和结果已保存到 {output_txt_path}")
    print(f"火势趋势分析已保存到 {trend_txt_path}")
    print(f"拟合前后的折线图已保存到 {gray_plot_path}")

# 主程序入口
if __name__ == "__main__":
    original_images_path = "red_channel_gray"
    masks_path = "exp2/masks/"
    output_txt_path = "exp2/results_gray_value_sums.txt"
    trend_txt_path = "exp2/fire_trend_analysis.txt"
    gray_plot_path = "exp2/red_channel_fire_trend_fitted_plot.png"

    process_images_and_judge_trend(original_images_path, masks_path, output_txt_path, trend_txt_path, gray_plot_path)
