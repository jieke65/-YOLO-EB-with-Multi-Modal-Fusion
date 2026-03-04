import cv2
import numpy as np

# 计算二值图像的形心
def compute_centroid(binary_image):
    # 获取图像的矩
    moments = cv2.moments(binary_image)
    # 计算形心坐标
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

# 加载图像
img1 = cv2.imread('l.png', cv2.IMREAD_GRAYSCALE)  # 第一张图像（灰度图）
img2 = cv2.imread('r.png', cv2.IMREAD_GRAYSCALE) # 第二张图像（灰度图）

# 假设二值图已经是二值化的，如果不是可以使用如下代码进行阈值处理
# _, img1_binary = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
# _, img2_binary = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# 计算第一张图像的形心
centroid1 = compute_centroid(img1)

# 计算第二张图像的形心
centroid2 = compute_centroid(img2)

# 创建ORB特征检测器
orb = cv2.ORB_create()

# 检测关键点和计算描述符
kp1, des1 = orb.detectAndCompute(img1, None)  # 第一张图像的关键点和描述符
kp2, des2 = orb.detectAndCompute(img2, None)  # 第二张图像的关键点和描述符

# 使用暴力匹配器（Brute-Force Matcher）进行描述符匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Hamming距离，交叉验证匹配
matches = bf.match(des1, des2)  # 匹配两张图像的描述符

# 按照匹配距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 在第一张图像上标出形心
img1_with_centroid = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)  # 将灰度图像转换为BGR，以便绘制
cv2.circle(img1_with_centroid, centroid1, 4, (0, 0, 255), -1)  # 用红色圆圈标出形心

# 在第二张图像上标出形心
img2_with_centroid = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)  # 将灰度图像转换为BGR，以便绘制
cv2.circle(img2_with_centroid, centroid2, 4, (0, 0, 255), -1)  # 用红色圆圈标出形心

# 绘制所有匹配点对
match_img_all = cv2.drawMatches(
    img1_with_centroid, kp1, img2_with_centroid, kp2, matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 显示结果（所有匹配点对）
cv2.imshow('All Matches with Centroids', match_img_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存所有匹配点对的结果图像
cv2.imwrite('all_matches_with_centroids.jpg', match_img_all)

# 绘制最优匹配点对
min_distance = float('inf')
best_match = None
for match in matches:
    # 获取第一张图像的匹配点
    pt1 = kp1[match.queryIdx].pt  # 该点在第一张图像中的位置
    # 计算该点到形心的欧几里得距离
    distance = np.sqrt((pt1[0] - centroid1[0])**2 + (pt1[1] - centroid1[1])**2)
    if distance < min_distance:
        min_distance = distance
        best_match = match

# 只保留最优匹配点对
best_matches = [best_match]

# 绘制最优匹配点对
match_img_best = cv2.drawMatches(
    img1_with_centroid, kp1, img2_with_centroid, kp2, best_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 显示最优匹配点对结果
cv2.imshow('Best Match with Centroids', match_img_best)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存最优匹配点对的结果图像
cv2.imwrite('best_match_with_centroids.jpg', match_img_best)
