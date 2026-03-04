import numpy as np  # 导入NumPy库用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import matplotlib.patches as mpatches  # 用于绘制图例中的色块

# 文件路径
input_file = '2.txt'  # 输入数据文件名
output_file = 'result.txt'  # 输出斜率和趋势结果的文件名

# 读取数据
intensities = []  # 创建空列表用于存储光谱强度值
with open(input_file, 'r') as file:  # 打开输入文件读取模式
    for line in file:  # 遍历每一行
        parts = line.strip().split()  # 去除空白字符并分割（按空格或制表符）
        if len(parts) == 2:  # 确保有两列数据
            intensities.append(float(parts[1]))  # 提取第二列作为强度值
intensities = np.array(intensities)  # 转换为NumPy数组以便后续计算

# 时间轴（每0.1秒一个数据点）
time = np.arange(0, len(intensities) * 0.1, 0.1)  # 生成时间序列，例如 [0.0, 0.1, 0.2, ...]

# 拟合参数
window_size = 3  # 每组数据的窗口大小，对应0.5秒（5个点）
slope_threshold = 0.05  # 斜率阈值，用于判断趋势（上升、下降或稳定）

trend_regions = []  # 用于保存每个拟合段的起止时间和对应颜色
slopes = []  # 用于保存每段的拟合斜率
slope_times = []  # 用于保存每段中点的时间（用于绘制斜率图）

# ---------- 保存斜率数据 ----------
with open(output_file, 'w') as f:  # 打开输出文件写入模式
    f.write("Start_Time(s)\tSlope\tTrend\n")  # 写入表头
    for i in range(0, len(intensities) - window_size + 1, window_size):  # 每隔window_size处理一段数据
        time_segment = time[i:i + window_size]  # 当前时间段
        intensity_segment = intensities[i:i + window_size]  # 当前强度段

        # 拟合直线 y = kx + b
        k, b = np.polyfit(time_segment, intensity_segment, 1)  # 用一阶多项式拟合数据段，返回斜率k和截距b
        fit_line = k * time_segment + b  # 计算拟合值（用于画线）

        # 根据斜率判断趋势
        if k > slope_threshold:
            trend = 'Rising'  # 上升
            color = 'lightgreen'  # 背景色为浅绿色
        elif k < -slope_threshold:
            trend = 'Falling'  # 下降
            color = 'lightcoral'  # 背景色为浅红色
        else:
            trend = 'Stable'  # 稳定
            color = 'lightgray'  # 背景色为灰色

        f.write(f"{time_segment[0]:.1f}\t{k:.4f}\t{trend}\n")  # 写入结果文件

        trend_regions.append((time_segment[0], time_segment[-1], color))  # 保存趋势区域
        slopes.append(k)  # 保存斜率
        slope_times.append(time_segment[0] + (time_segment[-1] - time_segment[0]) / 2)  # 保存中点时间

# ---------- 图1：原始数据 + 拟合趋势线 + 背景色 ----------
plt.figure(figsize=(14, 6))  # 创建图像，设定大小为14x6英寸

# 绘制背景色块
for start, end, color in trend_regions:
    plt.axvspan(start, end, facecolor=color, alpha=0.3)  # 在指定时间范围内填充颜色

# 绘制原始数据折线
plt.plot(time, intensities, color='black', linewidth=1.2, label='Original Data')  # 原始数据线

# 绘制拟合的趋势线段
for i in range(0, len(intensities) - window_size + 1, window_size):
    time_segment = time[i:i + window_size]  # 拟合段时间
    intensity_segment = intensities[i:i + window_size]  # 拟合段强度
    k, b = np.polyfit(time_segment, intensity_segment, 1)  # 拟合斜率与截距
    fit_line = k * time_segment + b  # 计算拟合值

    # 设置颜色（与趋势背景一致）
    if k > slope_threshold:
        color = 'green'
    elif k < -slope_threshold:
        color = 'red'
    else:
        color = 'gray'
    plt.plot(time_segment, fit_line, color=color, linewidth=2.0)  # 绘制趋势线段

# 设置图例和标签
legend_handles = [
    mpatches.Patch(color='lightgreen', label='Rising'),
    mpatches.Patch(color='lightcoral', label='Falling'),
    mpatches.Patch(color='lightgray', label='Stable')
]
plt.legend(handles=[*legend_handles, plt.Line2D([], [], color='black', label='Original Data')])
plt.xlabel('Time (s)')  # 设置X轴标签
plt.ylabel('Spectral Intensity')  # 设置Y轴标签
plt.title('Spectral Intensity with Fitted Trends')  # 图标题
plt.tight_layout()  # 自动调整布局
plt.grid(True)  # 添加网格
plt.savefig('trend_plot_with_fit.png', dpi=300)  # 保存为PNG图像，分辨率300dpi
plt.close()  # 关闭图像

# ---------- 图2：原始数据图（点 + 线 + 每点标注值） ----------
plt.figure(figsize=(12, 5))  # 新建图像窗口
plt.plot(time, intensities, color='blue', linewidth=1.2, label='Line')  # 原始光谱折线
plt.scatter(time, intensities, color='red', s=10, label='Points')  # 原始数据点
#
# # 添加每个点的强度值标注（每3个标注一个）
# for i, (t, y) in enumerate(zip(time, intensities)):
#     if i % 1 == 0:
#         plt.text(t, y + 1, f"{y:.1f}", fontsize=7, ha='center', va='bottom')

plt.xlabel('Time (s)')
plt.ylabel('Spectral Intensity')
plt.title('Original Spectral Intensity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('original_data_plot.png', dpi=300)  # 保存为PNG
plt.close()

# ---------- 图3：斜率变化图 ----------
plt.figure(figsize=(12, 4))  # 新建窗口
plt.plot(slope_times, slopes, marker='o', linestyle='-', color='purple')  # 绘制斜率随时间变化曲线
plt.axhline(slope_threshold, color='green', linestyle='--', label='Rising Threshold')  # 上升阈值线
plt.axhline(-slope_threshold, color='red', linestyle='--', label='Falling Threshold')  # 下降阈值线
plt.xlabel('Time (s)')
plt.ylabel('Slope (Intensity/sec)')
plt.title('Trend Slope Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('slope_plot.png', dpi=300)  # 保存为PNG图像
plt.close()
