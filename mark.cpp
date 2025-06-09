#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// 定义直线结构体
struct Line {
    Point pt1, pt2;
};

// 计算两点间欧氏距离
double distance(Point a, Point b) {
    return norm(a - b);
}

// 将线段转为 Ax + By + C = 0 的一般形式
void lineToGeneralForm(const Line& l, double& A, double& B, double& C) {
    A = l.pt2.y - l.pt1.y;
    B = l.pt1.x - l.pt2.x;
    C = l.pt2.x * l.pt1.y - l.pt1.x * l.pt2.y;
}

// 点到直线的距离
double pointToLineDistance(Point p, const Line& l) {
    double A, B, C;
    lineToGeneralForm(l, A, B, C);
    return fabs(A * p.x + B * p.y + C) / sqrt(A*A + B*B);
}

// 找到两条线段之间最近的一对点（垂线）
bool findClosestPointsBetweenLines(const Line& l1, const Line& l2, Point& p1, Point& p2) {
    vector<Point> candidates;
    candidates.push_back(l1.pt1);
    candidates.push_back(l1.pt2);
    candidates.push_back(l2.pt1);
    candidates.push_back(l2.pt2);

    double min_dist = numeric_limits<double>::max();
    for (const auto& p : candidates) {
        double d = pointToLineDistance(p, l2);
        if (d < min_dist) {
            min_dist = d;
            p1 = p;

            // 投影点
            double A, B, C;
            lineToGeneralForm(l2, A, B, C);
            double den = A*A + B*B;
            int x = (B*(B*p.x - A*p.y) - A*C) / den;
            int y = (A*(-B*p.x + A*p.y) - B*C) / den;
            p2 = Point(x, y);
        }
    }

    return true;
}

// 拟合线段为直线（Vec4f: (vx, vy, x0, y0) 表示方向+点）
Vec4f fitLineToSegment(const vector<Vec4i>& lines_segment) {
    vector<Point> points;
    for (const auto& line : lines_segment) {
        points.push_back(Point(line[0], line[1]));
        points.push_back(Point(line[2], line[3]));
    }

    Vec4f line_params;
    fitLine(points, line_params, DIST_L2, 0, 0.01, 0.01);
    return line_params;
}

// 计算两条直线之间的最短距离，并返回两个垂足点
bool getPerpendicularBetweenTwoLines(
        const Vec4f& l1, const Vec4f& l2,
        Point2f& p1, Point2f& p2)
{
    float x1 = l1[2], y1 = l1[3]; // 点在线上
    float dx1 = l1[0], dy1 = l1[1]; // 方向向量

    float x2 = l2[2], y2 = l2[3];
    float dx2 = l2[0], dy2 = l2[1];

    float cross = dx1 * dy2 - dx2 * dy1;

    if (fabs(cross) < 1e-6) { // 平行
        return false;
    }

    float t1 = ((x2 - x1)*dy2 - (y2 - y1)*dx2) / cross;
    float t2 = ((x2 - x1)*dy1 - (y2 - y1)*dx1) / cross;

    p1 = Point2f(x1 + t1 * dx1, y1 + t1 * dy1);
    p2 = Point2f(x2 + t2 * dx2, y2 + t2 * dy2);

    return true;
}

void drawExtendedLine(Mat& img, const Vec4f& line, Scalar color, int thickness = 2) {
    float vx = line[0], vy = line[1];
    float x0 = line[2], y0 = line[3];

    float length = max(img.rows, img.cols); // 足够长的长度

    Point pt1(cvRound(x0 + length * vx), cvRound(y0 + length * vy));
    Point pt2(cvRound(x0 - length * vx), cvRound(y0 - length * vy));

    // 裁剪到图像范围内
    Rect rect(0, 0, img.cols, img.rows);
    clipLine(rect, pt1, pt2);

    cv::line(img, pt1, pt2, color, thickness, LINE_AA);
}

int main() {
    // 读取图像
    Mat image = imread("/home/HwHiAiUser/ocr/pic/ocr_water.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return -1;
    }

    Mat result = image.clone();

    // 转换为HSV色彩空间
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // 定义红色范围（两个区间）
    Scalar lower_red1(0, 100, 100);
    Scalar upper_red1(10, 255, 255);
    Scalar lower_red2(160, 100, 100);
    Scalar upper_red2(180, 255, 255);

    Mat mask1, mask2, red_mask;
    inRange(hsv, lower_red1, upper_red1, mask1);
    inRange(hsv, lower_red2, upper_red2, mask2);
    red_mask = mask1 | mask2;

    // 形态学操作连接断开的线
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    // 边缘检测
    Mat edges;
    Canny(red_mask, edges, 50, 150);

    // 霍夫变换提取线段
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);

    vector<Vec4i> filtered_lines;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1];
        int x2 = line[2], y2 = line[3];

        double dx = x2 - x1;
        double dy = y2 - y1;
        double slope = dy / (dx + 1e-6);
        double angle = abs(atan(slope) * 180 / CV_PI);

        if ((angle <= 30) || (angle >= 150)) {
            filtered_lines.push_back(line);
        }
    }

    // 存储所有线段的中心 y 值
    vector<double> center_ys;
    for (const auto& line : filtered_lines) {
        double mid_y = (line[1] + line[3]) / 2.0;
        center_ys.push_back(mid_y);
    }

    // 排序 y 值
    sort(center_ys.begin(), center_ys.end());

    // 使用 KMeans 聚类找出最可能的 4 条线
    vector<double> cluster_centers;
    if (center_ys.size() >= 4) {
        // 初始聚类中心：均匀选取
        cluster_centers = {center_ys[0], center_ys[center_ys.size()/3],
                           center_ys[2*center_ys.size()/3], center_ys.back()};

        for (int iter = 0; iter < 10; ++iter) {
            vector<vector<double>> clusters(4);
            for (double y : center_ys) {
                int closest = 0;
                double min_dist = abs(y - cluster_centers[0]);
                for (int i = 1; i < 4; ++i) {
                    double d = abs(y - cluster_centers[i]);
                    if (d < min_dist) {
                        min_dist = d;
                        closest = i;
                    }
                }
                clusters[closest].push_back(y);
            }

            // 更新聚类中心
            for (int i = 0; i < 4; ++i) {
                if (!clusters[i].empty()) {
                    double sum = accumulate(clusters[i].begin(), clusters[i].end(), 0.0);
                    cluster_centers[i] = sum / clusters[i].size();
                }
            }
        }

        // 排序聚类中心
        sort(cluster_centers.begin(), cluster_centers.end());

        // 绘制原始线段（绿色）
        for (const auto& line : lines) {
            cv::line(result, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 255, 0), 2);
        }

        cv::imwrite("/home/HwHiAiUser/ocr/debug/line.jpg",result);

        // 选择与每个 cluster_center 最接近的线段作为代表
        vector<Line> selected_lines;
        for (double target_y : cluster_centers) {
            double min_diff = numeric_limits<double>::max();
            Vec4i best_line;
            for (const auto& line : lines) {
                double mid_y = (line[1] + line[3]) / 2.0;
                double diff = abs(mid_y - target_y);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_line = line;
                }
            }
            selected_lines.push_back({Point(best_line[0], best_line[1]), Point(best_line[2], best_line[3])});
        }

        // 绘制选中的 4 条线（不同颜色区分）
        vector<Scalar> colors = {Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(0, 255, 255), Scalar(255, 255, 0)};
        for (int i = 0; i < selected_lines.size(); ++i) {
            const auto& l = selected_lines[i];
            line(result, l.pt1, l.pt2, colors[i], 2);
        }

        // 存储所有线段对应的直线参数
        vector<vector<Vec4i>> line_groups(4); // 每组线段对应一个聚类
        int group_idx = 0; // 初始化聚类索引
        // 在选择代表线段时，保存每个聚类的所有线段
        for (double target_y : cluster_centers) {
            vector<Vec4i> group;
            for (const auto& line : filtered_lines) {
                double mid_y = (line[1] + line[3]) / 2.0;
                if (abs(mid_y - target_y) < 30) { // 控制聚类半径
                    group.push_back(line);
                }
            }

            if (!group.empty()) {
                line_groups[group_idx++] = group;
            } else {
                cout << "Warning: Cluster " << group_idx << " has no matching lines!" << endl;
            }
        }

        vector<Vec4f> fitted_lines;

        for (int i = 0; i < group_idx; ++i) {
            Vec4f line = fitLineToSegment(line_groups[i]);
            fitted_lines.push_back(line);
        }

        // 使用不同颜色加粗绘制拟合后的直线
        vector<Scalar> line_colors = {Scalar(0, 0, 255), Scalar(255, 0, 0),
                                      Scalar(0, 255, 255), Scalar(255, 255, 0)};

        for (int i = 0; i < fitted_lines.size(); ++i) {
            drawExtendedLine(result, fitted_lines[i], line_colors[i], 3); // 加粗显示
        }

        vector<double> distances;

        for (int i = 0; i < fitted_lines.size() - 1; ++i) {
            Point2f p1, p2;
            if (getPerpendicularBetweenTwoLines(fitted_lines[i], fitted_lines[i + 1], p1, p2)) {

                // 裁剪到图像范围内
                bool in_bounds = (p1.x >= 0 && p1.y >= 0 &&
                                  p1.x < image.cols && p1.y < image.rows &&
                                  p2.x >= 0 && p2.y >= 0 &&
                                  p2.x < image.cols && p2.y < image.rows);

                if (in_bounds) {
                    line(result, p1, p2, Scalar(0, 0, 255), 1, LINE_AA);

                    double dist = distance(p1, p2);
                    distances.push_back(dist);

                    string label = to_string(static_cast<int>(dist)) + " px";
                    putText(result, label, Point((p1.x + p2.x)/2 + 5, (p1.y + p2.y)/2),
                            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);

                    cout << "Distance between line " << i << " and " << i+1 << ": " << dist << " px" << endl;
                } else {
                    cout << "垂足超出图像范围，跳过。" << endl;
                }
            } else {
                cout << "线段平行或无法计算垂线。" << endl;
            }
        }

        // 绘制相邻线之间的垂线（基于拟合直线）
//        vector<double> distances;

        for (int i = 0; i < 3; ++i) {
            // 拟合直线
            Vec4f line1 = fitLineToSegment(line_groups[i]);
            Vec4f line2 = fitLineToSegment(line_groups[i + 1]);

            Point2f p1, p2;
            if (getPerpendicularBetweenTwoLines(line1, line2, p1, p2)) {

                // 限制垂足在图像范围内
                if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows &&
                    p2.x >= 0 && p2.x < image.cols && p2.y >= 0 && p2.y < image.rows) {

                    line(result, p1, p2, Scalar(0, 0, 255), 1, LINE_AA);

                    double dist = distance(p1, p2);
                    distances.push_back(dist);

                    string label = to_string(static_cast<int>(dist)) + " px";
                    putText(result, label, Point((p1.x + p2.x)/2 + 5, (p1.y + p2.y)/2),
                            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
                }
            }
        }

        // 平均像素距离 & 比例因子
        double avg_pixel_distance = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
        cout << "Average pixel distance between lines: " << avg_pixel_distance << " px" << endl;

        double real_distance = 0.5; // 相邻线真实距离单位
        double ratio = real_distance / avg_pixel_distance;
        cout << "Pixel to real-world scale ratio: " << ratio << " unit/px" << endl;

        // 显示和保存结果
        imwrite("/home/HwHiAiUser/ocr/debug/det_line.png", result);

    } else {
        cout << "Not enough lines detected for clustering." << endl;
    }

    return 0;
}