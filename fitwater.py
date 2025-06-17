import cv2
import numpy as np


#=====================================================生成水面掩码=====================================================
# 读取第二张图（包含水面）
img_water = cv2.imread("fitwater.png")

# 转换为HSV空间便于分离黑色区域
hsv = cv2.cvtColor(img_water, cv2.COLOR_BGR2HSV)

# 定义黑色范围 (V通道值低)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 50, 40]) 

# 创建水面掩码
mask = cv2.inRange(hsv, lower_black, upper_black)

# 形态学操作优化掩码
kernel = np.ones((7,7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=2)  # 填充小孔洞
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)   # 移除小噪点

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
max_area = 0
main_label = 1

# 寻找最大连通域（水面主体）
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > max_area:
        max_area = area
        main_label = i

clean_mask = np.zeros_like(mask)
clean_mask[labels == main_label] = 255

# 裁剪顶部干扰（假设水面在图像下半部）
height, width = clean_mask.shape
clean_mask[0:int(height*0.25), :] = 0 

# 保存水面掩码(测试)
cv2.imwrite("./water_mask.png", clean_mask)


#=====================================================计算斜率=====================================================
# 读取第一张图（包含尺度线）
img_scale = cv2.imread("fitwater.png")

# 预处理
gray = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# 计算所有直线的平均斜率
slopes = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x2 - x1) > 10:  # 避免垂直线
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

# 取中值作为参考斜率
k_ref = np.median(slopes)
print(f"检测到尺度线斜率: {k_ref:.4f}")

#=====================拟合=========================================
# 提取水面上边界点
boundary_points = []
height, width = clean_mask.shape
for x in range(width):
    col = clean_mask[:, x]  # 使用过滤后的掩码
    y_positions = np.where(col == 255)[0]
    if len(y_positions) > 0:
        # 改进6：添加高度过滤，防止提取顶部点
        y_top = np.min(y_positions)
        if y_top > height * 0.3:  # 只取图像高度30%以下的点
            boundary_points.append([x, y_top]) 

# 转换为NumPy数组
points = np.array(boundary_points)

if len(points) > 0:
    median_y = np.median(points[:, 1])
    points = points[points[:, 1] < median_y + 50]  # 只保留上半部分点

# 拟合水面线 (固定斜率k_ref)
x_vals = points[:, 0]
y_vals = points[:, 1]
b = np.mean(y_vals - k_ref * x_vals)  # 最小二乘解

# 在原图上绘制结果
result_img = img_water.copy()
y_start = int(k_ref * 0 + b)
y_end = int(k_ref * result_img.shape[1] + b)
cv2.line(result_img, (0, y_start), (result_img.shape[1], y_end), (0, 255, 0), 2)

# 添加可视化点（调试用）
for pt in points:
    cv2.circle(result_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)  # 红色点标记边界点

# 保存并显示
cv2.imwrite("./result_with_line.png", result_img)
