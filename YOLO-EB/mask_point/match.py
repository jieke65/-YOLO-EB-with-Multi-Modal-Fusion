import cv2
import os
import sys
import cv2
from ctypes import *
import numpy as np
import configparser
import scipy.io
def load_camera_para_from_mat(para_file):# 定义一个函数，用于从.mat文件中加载相机参数
    try:
        params = scipy.io.loadmat(para_file)#用scipy.io.loadmat函数加载.mat文件中的参数
        matrix0 = params['cameraMatrix1']#提取左相机的内参矩阵
        dist0 = params['distCoeffs1']#提取左相机的畸变系数
        matrix1 = params['cameraMatrix2']#提取右相机的内参矩阵
        dist1 = params['distCoeffs2']#提取右相机的畸变系数
        R = params['R']#提取旋转矩阵R
        T = params['T']#提取平移向量T
    except Exception as exception:
        print("Error:", exception)# 如果在加载过程中发生任何异常，打印错误信息
    return matrix0, dist0, matrix1, dist1, R, T#返回相机参数

cam_matrix1, cam_dist1, cam_matrix2, cam_dist2, cam_R, cam_T = load_camera_para_from_mat('CAM_PARAMETER.mat')
cam_T=cam_T.T # 转置平移向量矩阵

# 假设已经计算出这些参数

B = np.linalg.norm(cam_T)# 基线长度（单位：毫米）

disparity = 176#视差值（单位：像素）

# 设置路径
left_images_path = 'data/images/left'
right_images_path = 'data/images/right'
left_masks_path = 'runs/detect/exp/masks/left'
right_masks_path = 'runs/detect/exp/masks/right'
seg_left_path = 'runs/detect/exp/seg/left'
seg_right_path = 'runs/detect/exp/seg/right'
seg_max_left_path = 'runs/detect/exp/seg_max/left'
seg_max_right_path = 'runs/detect/exp/seg_max/right'
matches_path = 'runs/detect/exp/matches'
best_match_path = 'runs/detect/exp/best_match'
output_best_match_txt = 'runs/detect/exp/best_match_coordinates.txt'  # 最优匹配坐标文件
output_max_com_txt = 'runs/detect/exp/max_com_coordinates.txt'  # 最大掩膜形心坐标文件

# 创建保存目录
os.makedirs(seg_left_path, exist_ok=True)
os.makedirs(seg_right_path, exist_ok=True)
os.makedirs(seg_max_left_path, exist_ok=True)
os.makedirs(seg_max_right_path, exist_ok=True)
os.makedirs(matches_path, exist_ok=True)
os.makedirs(best_match_path, exist_ok=True)

# 获取图片和掩膜文件列表
left_images = sorted(os.listdir(left_images_path))
right_images = sorted(os.listdir(right_images_path))
left_masks = sorted(os.listdir(left_masks_path))
right_masks = sorted(os.listdir(right_masks_path))

# 特征匹配设置
orb = cv2.ORB_create()




# 提取相机参数
fx = cam_matrix1[0, 0]
fy = cam_matrix1[1, 1]
cx =cam_matrix1[0, 2]
cy = cam_matrix1[1, 2]
B = np.linalg.norm(cam_T)# 基线长度（单位：毫米）

# 创建输出的txt文件并写入标题
with open(output_best_match_txt, 'w') as f:
    f.write("Image_Name:left_x, left_y, right_x, right_y, disparity, X, Y, Z\n")

with open(output_max_com_txt, 'w') as f:
    f.write("Image_Name:left_com_x, left_com_y, right_com_x, right_com_y, disparity, X, Y, Z\n")

# 遍历左图和右图
for left_img_name, right_img_name in zip(left_images, right_images):
    # 读取原图
    left_img = cv2.imread(os.path.join(left_images_path, left_img_name))
    right_img = cv2.imread(os.path.join(right_images_path, right_img_name))

    # 提取原图编号（假设格式为 1.jpg, 2.jpg, ...）
    left_img_id = left_img_name.split('.')[0]  # 提取编号（例如：1，2）
    right_img_id = right_img_name.split('.')[0]  # 提取编号（例如：1，2）

    # 根据原图编号筛选对应的掩膜图
    left_mask_names = [name for name in left_masks if name.startswith(f"{left_img_id}_mask")]
    right_mask_names = [name for name in right_masks if name.startswith(f"{right_img_id}_mask")]

    # 最大掩膜选择变量
    max_left_mask_area = 0
    max_left_mask = None
    max_right_mask_area = 0
    max_right_mask = None
    left_center_of_mass = None
    right_center_of_mass = None
    best_left_com = None  # 用于存储最佳匹配点的左图形心
    best_right_com = None  # 用于存储最佳匹配点的右图形心

    # 处理左图的掩膜图
    for left_mask_name in left_mask_names:
        # 读取掩膜并调整大小
        left_mask = cv2.imread(os.path.join(left_masks_path, left_mask_name), cv2.IMREAD_GRAYSCALE)
        left_mask_resized = cv2.resize(left_mask, (left_img.shape[1], left_img.shape[0]))

        # 计算掩膜区域的面积
        mask_area = np.sum(left_mask_resized == 255)

        # 寻找最大掩膜区域
        if mask_area > max_left_mask_area:
            max_left_mask_area = mask_area
            max_left_mask = left_mask_resized
            # 计算最大掩膜的形心
            moments = cv2.moments(left_mask_resized)
            if moments["m00"] != 0:
                left_center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                print(f"Shape center of left mass for {left_mask_name}: {left_center_of_mass}")
        # 创建黑色图像
        left_img_masked = np.zeros_like(left_img)
        # 将掩膜区域保留，其他区域设置为黑色
        left_img_masked[left_mask_resized == 255] = left_img[left_mask_resized == 255]

        # 保存所有掩膜区域结果
        cv2.imwrite(os.path.join(seg_left_path, left_img_name), left_img_masked)

    # 处理右图的掩膜图
    for right_mask_name in right_mask_names:
        # 读取掩膜并调整大小
        right_mask = cv2.imread(os.path.join(right_masks_path, right_mask_name), cv2.IMREAD_GRAYSCALE)
        right_mask_resized = cv2.resize(right_mask, (right_img.shape[1], right_img.shape[0]))

        # 计算掩膜区域的面积
        mask_area = np.sum(right_mask_resized == 255)

        # 寻找最大掩膜区域
        if mask_area > max_right_mask_area:
            max_right_mask_area = mask_area
            max_right_mask = right_mask_resized
            # 计算最大掩膜的形心
            moments = cv2.moments(right_mask_resized)
            if moments["m00"] != 0:
                right_center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                print(f"Shape center of right mass for {right_mask_name}: {right_center_of_mass}")
        # 创建黑色图像
        right_img_masked = np.zeros_like(right_img)
        # 将掩膜区域保留，其他区域设置为黑色
        right_img_masked[right_mask_resized == 255] = right_img[right_mask_resized == 255]

        # 保存所有掩膜区域结果
        cv2.imwrite(os.path.join(seg_right_path, right_img_name), right_img_masked)

    # 计算最佳匹配点（这里假设我们通过ORB特征匹配来获取最佳匹配点）
    # 使用ORB特征匹配来计算最佳匹配点坐标
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_img, None)
    kp2, des2 = orb.detectAndCompute(right_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if matches:
        best_match = min(matches, key=lambda x: x.distance)
        best_left_com = (int(kp1[best_match.queryIdx].pt[0]), int(kp1[best_match.queryIdx].pt[1]))
        best_right_com = (int(kp2[best_match.trainIdx].pt[0]), int(kp2[best_match.trainIdx].pt[1]))
        print(f"Best match coordinates: left: {best_left_com}, right: {best_right_com}")

    # 如果找到最大掩膜区域，则更新原图只保留最大区域，其他区域设为黑色
    if max_left_mask is not None:
        # 创建新的黑色图像
        left_img_max_masked = np.zeros_like(left_img)
        # 只保留最大掩膜区域
        left_img_max_masked[max_left_mask == 255] = left_img[max_left_mask == 255]
        # 标记形心
        if left_center_of_mass:
            cv2.circle(left_img_max_masked, left_center_of_mass, 0, (0, 0, 255), -1)
        # 保存结果
        cv2.imwrite(os.path.join(seg_max_left_path, left_img_name), left_img_max_masked)

    if max_right_mask is not None:
        # 创建新的黑色图像
        right_img_max_masked = np.zeros_like(right_img)
        # 只保留最大掩膜区域
        right_img_max_masked[max_right_mask == 255] = right_img[max_right_mask == 255]
        # 标记形心
        if right_center_of_mass:
            cv2.circle(right_img_max_masked, right_center_of_mass, 0, (0, 0, 255), -1)
        # 保存结果
        cv2.imwrite(os.path.join(seg_max_right_path, right_img_name), right_img_max_masked)

    # 计算视差和三维坐标
    if best_left_com and best_right_com:  # 使用最佳匹配点计算视差和三维坐标
        disparity = best_left_com[0] - best_right_com[0]  # 视差 = 左右图像X坐标差
    else:
        disparity = left_center_of_mass[0] - right_center_of_mass[0]  # 视差 = 左右图像X坐标差

    if disparity != 0:
        Z = (fx * B) / disparity
        X = (left_center_of_mass[0] - cx) * Z / fx
        Y = -(left_center_of_mass[1] - cy) * Z / fy
    else:
        Z = X = Y = 0  # 防止除以零

    # 将最优匹配点坐标、视差、三维坐标保存到txt文件
    with open(output_best_match_txt, 'a') as f:
        f.write(
            f"{left_img_name}: {best_left_com[0]}, {best_left_com[1]}, {best_right_com[0]}, {best_right_com[1]}, {disparity:.2f}, {X:.2f}, {Y:.2f}, {Z:.2f}\n")

    # 将最大掩膜形心坐标、视差、三维坐标保存到txt文件
    with open(output_max_com_txt, 'a') as f:
        f.write(
            f"{left_img_name}: {left_center_of_mass[0]}, {left_center_of_mass[1]}, {right_center_of_mass[0]}, {right_center_of_mass[1]}, {disparity:.2f}, {X:.2f}, {Y:.2f}, {Z:.2f}\n")

print("处理完成，结果已保存到对应目录。")

