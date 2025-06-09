#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

// 定义拟合直线结构体
struct FittedLine {
    Point p1;
    Point p2;
    double k;
    double b;
};

int main() {
    // 1. 读取图片
    Mat img = imread("/home/HwHiAiUser/ocr/pic/ocr_water.png");
    if (img.empty()) {
        cerr << "错误：找不到图片文件 ruler.jpg" << endl;
        return -1;
    }

    // 2. 转换到HSV色彩空间
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 3. 红色范围遮罩
    Mat mask1, mask2, mask;
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
    bitwise_or(mask1, mask2, mask);

    // 4. 形态学操作去噪
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    // 5. 边缘检测
    Mat edges;
    Canny(mask, edges, 30, 90, 3);

    // 6. 霍夫直线检测
    vector<Vec4i> linesP;
    HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 60, 15);

    // 7. 过滤水平线段
    vector<Vec4i> horizontalLines;
    for (const Vec4i& line : linesP) {
        Point pt1(line[0], line[1]);
        Point pt2(line[2], line[3]);
        double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180 / CV_PI;
        
        if (abs(angle) < 30) {
            horizontalLines.push_back(line);
        }
    }

    // 8. 合并相似线段
    vector<FittedLine> fittedLines;
    const int y_threshold = 40;
    vector<vector<Point>> groups;
    vector<double> groupYAvg;
    vector<int> groupCount;

    for (const Vec4i& line : horizontalLines) {
        Point pt1(line[0], line[1]);
        Point pt2(line[2], line[3]);
        double y_avg = (pt1.y + pt2.y) / 2.0;
        
        bool foundGroup = false;
        for (size_t i = 0; i < groups.size(); ++i) {
            if (abs(y_avg - groupYAvg[i]) < y_threshold) {
                groups[i].push_back(pt1);
                groups[i].push_back(pt2);
                groupYAvg[i] = (groupYAvg[i] * groupCount[i] + y_avg) / (groupCount[i] + 1);
                groupCount[i]++;
                foundGroup = true;
                break;
            }
        }
        
        if (!foundGroup) {
            groups.push_back({pt1, pt2});
            groupYAvg.push_back(y_avg);
            groupCount.push_back(1);
        }
    }

    // 9. 对每组点进行线性拟合
    for (const auto& group : groups) {
        if (group.size() < 2) continue;
        
        vector<double> xs, ys;
        for (const Point& p : group) {
            xs.push_back(p.x);
            ys.push_back(p.y);
        }
        
        // 最小二乘拟合
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = xs.size();
        
        for (int i = 0; i < n; ++i) {
            sumX += xs[i];
            sumY += ys[i];
            sumXY += xs[i] * ys[i];
            sumX2 += xs[i] * xs[i];
        }
        
        double k = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double b = (sumY - k * sumX) / n;
        
        // 计算线段端点
        auto [minX, maxX] = minmax_element(xs.begin(), xs.end());
        Point p1(*minX, static_cast<int>(k * *minX + b));
        Point p2(*maxX, static_cast<int>(k * *maxX + b));
        
        fittedLines.push_back({p1, p2, k, b});
    }

    // 10. 按y坐标排序
    sort(fittedLines.begin(), fittedLines.end(), 
        [](const FittedLine& a, const FittedLine& b) {
            return (a.p1.y + a.p2.y) < (b.p1.y + b.p2.y);
        });

    // 11. 计算并标注距离
    vector<double> distances;
    for (size_t i = 1; i < fittedLines.size(); ++i) {
        // 当前线段中点
        Point center(
            (fittedLines[i].p1.x + fittedLines[i].p2.x) / 2,
            (fittedLines[i].p1.y + fittedLines[i].p2.y) / 2);
        
        // 前一线段参数
        double k_prev = fittedLines[i-1].k;
        double b_prev = fittedLines[i-1].b;
        
        // 计算垂足
        double x_int = (center.x + k_prev * (center.y - b_prev)) / (k_prev * k_prev + 1);
        double y_int = k_prev * x_int + b_prev;
        Point foot(static_cast<int>(x_int), static_cast<int>(y_int));
        
        // 计算距离
        double dist = sqrt(pow(center.x - x_int, 2) + pow(center.y - y_int, 2));
        distances.push_back(dist);
        
        // 绘制垂线
        line(img, center, foot, Scalar(255, 0, 0), 5);
        
        // 标注距离
        Point textPos(
            (center.x + foot.x) / 2 + 10,
            (center.y + foot.y) / 2);
        putText(img, format("%.1f", dist), textPos, 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    // 12. 输出结果
    cout << "检测到的红色刻度线数量: " << fittedLines.size() << endl;
    for (size_t i = 0; i < fittedLines.size(); ++i) {
        cout << "第" << i+1 << "条线: ("
             << fittedLines[i].p1.x << "," << fittedLines[i].p1.y << ") - ("
             << fittedLines[i].p2.x << "," << fittedLines[i].p2.y << ")" << endl;
    }

    cout << "相邻红色刻度线的像素距离:" << endl;
    for (size_t i = 0; i < distances.size(); ++i) {
        cout << "第" << i+1 << "与第" << i+2 << "条线之间: "
             << fixed << setprecision(2) << distances[i] << " 像素" << endl;
    }

    // 13. 可视化结果
    for (const FittedLine& line : fittedLines) {
        cv::line(img, line.p1, line.p2, Scalar(0, 255, 0), 2);
    }

    // 14. 显示结果
//    namedWindow("Result", WINDOW_NORMAL);
//    imshow("Result", img);
//    waitKey(0);
    cv::imwrite("./narked_ruler_cpp.jpg",img);

    return 0;
}