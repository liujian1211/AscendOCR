import cv2
import numpy as np

# 读取图片
img = cv2.imread('ocr_water.png')
if img is None:
    raise FileNotFoundError("找不到图片文件 ruler.jpg")

# 转换到HSV色彩空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 红色范围遮罩（红色有两个区间）
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 形态学操作去噪
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 边缘检测
edges = cv2.Canny(mask, 30, 90, apertureSize=3)

# 霍夫直线检测（调整参数）
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=50,         # 增大阈值
    minLineLength=60,     # 增大最小线段长度
    maxLineGap=15          # 减小最大间隙
)

# 过滤大致为“横向”的红色线段
horizontal_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # 只保留角度在-30到+30度之间的线段（可根据实际调整）
        if -30 < angle < 30:
            horizontal_lines.append(((x1, y1), (x2, y2)))

merged_lines = []
merged_groups = []
y_threshold = 40  # y坐标合并阈值，可根据实际调整

for line in horizontal_lines:
    (x1, y1), (x2, y2) = line
    y_avg = (y1 + y2) / 2
    found = False
    for group in merged_groups:
        if abs(y_avg - group['y_avg']) < y_threshold:
            group['points'].extend([(x1, y1), (x2, y2)])
            group['y_avg'] = (group['y_avg'] * group['count'] + y_avg) / (group['count'] + 1)
            group['count'] += 1
            found = True
            break
    if not found:
        merged_groups.append({'points': [(x1, y1), (x2, y2)], 'y_avg': y_avg, 'count': 1})

horizontal_lines = []
for group in merged_groups:
    pts = np.array(group['points'])
    xs = pts[:, 0]
    ys = pts[:, 1]
    if len(xs) >= 2:
        # 拟合一次直线 y = kx + b
        k, b = np.polyfit(xs, ys, 1)
        min_x = int(xs.min())
        max_x = int(xs.max())
        y1_fit = int(k * min_x + b)
        y2_fit = int(k * max_x + b)
        horizontal_lines.append(((min_x, y1_fit), (max_x, y2_fit)))

# 按y坐标排序
horizontal_lines.sort(key=lambda l: (l[0][1] + l[1][1]) / 2)


# 计算相邻线的像素距离
distances = []
for i in range(1, len(horizontal_lines)):
    # 取第i条线的中点
    x0 = int((horizontal_lines[i][0][0] + horizontal_lines[i][1][0]) / 2)
    y0 = int((horizontal_lines[i][0][1] + horizontal_lines[i][1][1]) / 2)
    # 上一条线的拟合参数
    x1, y1 = horizontal_lines[i-1][0]
    x2, y2 = horizontal_lines[i-1][1]
    if x2 != x1:
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # 点到直线距离
        dist = abs(k * x0 - y0 + b) / np.sqrt(k**2 + 1)
    else:
        # 垂直线特殊处理
        dist = abs(x0 - x1)
    distances.append(dist)
    # 绘制垂足
    if x2 != x1:
        # 计算垂足坐标
        k_perp = -1 / k if k != 0 else 0
        b_perp = y0 - k_perp * x0
        # 联立方程求交点
        x_int = (b_perp - b) / (k - k_perp) if (k - k_perp) != 0 else x0
        y_int = k * x_int + b
        cv2.line(img, (x0, y0), (int(x_int), int(y_int)), (255, 0, 0), 5)
        # 标注距离
        text_x = int((x0 + x_int) / 2) + 10
        text_y = int((y0 + y_int) / 2)
        cv2.putText(img, f"{dist:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # 垂直线直接连
        cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 5)
        text_x = int((x0 + x1) / 2) + 10
        text_y = int((y0 + y1) / 2)
        cv2.putText(img, f"{dist:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 输出结果
print("检测到的红色刻度线数量:", len(horizontal_lines))
for idx, ((x1, y1), (x2, y2)) in enumerate(horizontal_lines):
    print(f"第{idx+1}条线: ({x1},{y1}) - ({x2},{y2})")

print("相邻红色刻度线的像素距离:")
for i, d in enumerate(distances):
    print(f"第{i+1}与第{i+2}条线之间: {d:.2f} 像素")

# 可视化结果
for (x1, y1), (x2, y2) in horizontal_lines:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


#绘制一条水位线（这里替代为你们的水位线检测代码）
x1, y1 = horizontal_lines[-1][0]
x2, y2 = horizontal_lines[-1][1]

shift = 70
cv2.line(img, (x1, y1 + shift), (x2, y2 + shift), (0, 165, 255), 5)  # 橙色

wx1 = x1
wy1 = y1+shift
wx2=x2
wy2=y2+shift

mx = int((wx1 + wx2) / 2)
my = int((wy1 + wy2) / 2)

# 计算最后一条红色水尺线的参数
if x2 != x1:
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    # 点到直线距离
    dist = abs(k * mx - my + b) / np.sqrt(k**2 + 1)
    # 计算垂足坐标
    k_perp = -1 / k if k != 0 else 0
    b_perp = my - k_perp * mx
    x_int = (b_perp - b) / (k - k_perp) if (k - k_perp) != 0 else mx
    y_int = k * x_int + b
    # 绘制垂线
    cv2.line(img, (mx, my), (int(x_int), int(y_int)), (255, 0, 255), 4)
    # 标注距离
    text_x = int((mx + x_int) / 2) + 10
    text_y = int((my + y_int) / 2)
    cv2.putText(img, f"{dist:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
else:
    dist = abs(mx - x1)
    cv2.line(img, (mx, my), (x1, y1), (255, 0, 255), 4)
    text_x = int((mx + x1) / 2) + 10
    text_y = int((my + y1) / 2)
    cv2.putText(img, f"{dist:.1f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

cv2.imwrite('./marked_ruler.jpg', img)
print("已将检测结果保存为 marked_ruler.jpg")
#图里的像素距离是66，乘以比率0.004166667，等于0.275，用上一个水位深度减去该值为：1060.5-0.275=1060.225,实际水位应该是1060.200，误差不大




