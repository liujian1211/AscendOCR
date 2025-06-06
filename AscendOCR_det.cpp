#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "acl/acl.h"

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)

using namespace cv;
using namespace std;

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

namespace {
    const int det_input_size = 736;
    const float det_threshold = 0.3f;
    const float box_threshold = 0.5f;
    const float unclip_ratio = 1.4f;
}

class AscendOCR {
public:
    AscendOCR(const char* det_model_path);
    ~AscendOCR();
    Result Init();
    Result ProcessImage(const string& image_path);
    Result DetectText();
    vector<vector<Point2f>> GetBoxes() { return dt_boxes_; }
    void VisualizeResults(const string& output_path);

private:
    void ReleaseResource();
    void PostprocessDet(float* output_data, size_t output_size);
    void OrderPointsClockwise(vector<Point2f>& box);
    vector<Point2f> UnclipBox(const vector<Point2f>& box);
    float BoxScoreFast(Mat& bitmap, const vector<Point2f>& box);
    void FilterBoxes();

    // 模型相关变量
    aclrtContext context_;
    aclrtStream stream_;
    aclrtRunMode run_mode_;

    // 检测模型
    uint32_t det_model_id_;
    aclmdlDesc* det_model_desc_;
    aclmdlDataset* det_input_dataset_;
    aclmdlDataset* det_output_dataset_;
    void* det_input_buffer_;
    size_t det_input_buffer_size_;
    size_t det_output_size_;

    // 图像处理相关
    Mat src_image_;
    vector<vector<Point2f>> dt_boxes_;

    // 模型路径
    const char* det_model_path_;

    void PrepareInput();

    float scale_x_;
    float scale_y_;

    vector<Point2f> TransformBoxToOriginal(const vector<Point2f>& box);
};

AscendOCR::AscendOCR(const char* det_model_path)
        : det_model_path_(det_model_path),
          context_(nullptr), stream_(nullptr),
          det_model_id_(0), det_model_desc_(nullptr),
          det_input_dataset_(nullptr), det_output_dataset_(nullptr),
          det_input_buffer_(nullptr), det_output_size_(0) {}

AscendOCR::~AscendOCR() {
    ReleaseResource();
}

Result AscendOCR::Init() {
    // 初始化ACL环境
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtSetDevice(0); // 使用设备0
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateContext(&context_, 0);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&run_mode_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtGetRunMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // 加载检测模型
    ret = aclmdlLoadFromFile(det_model_path_, &det_model_id_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Load det model failed, errorCode is %d", ret);
        return FAILED;
    }

    det_model_desc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(det_model_desc_, det_model_id_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Get det model desc failed, errorCode is %d", ret);
        return FAILED;
    }

    // 创建检测模型输入输出数据集
    det_input_dataset_ = aclmdlCreateDataset();
    det_output_dataset_ = aclmdlCreateDataset();

    // 获取输入缓冲区
    size_t det_input_index = 0;
    det_input_buffer_size_ = aclmdlGetInputSizeByIndex(det_model_desc_, det_input_index);
    ret = aclrtMalloc(&det_input_buffer_, det_input_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Malloc det input buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    aclDataBuffer* det_input_data = aclCreateDataBuffer(det_input_buffer_, det_input_buffer_size_);
    ret = aclmdlAddDatasetBuffer(det_input_dataset_, det_input_data);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Add det input dataset buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    // 获取输出缓冲区
    size_t det_output_index = 0;
    det_output_size_ = aclmdlGetOutputSizeByIndex(det_model_desc_, det_output_index);
    void* det_output_buffer;
    ret = aclrtMalloc(&det_output_buffer, det_output_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Malloc det output buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    aclDataBuffer* det_output_data = aclCreateDataBuffer(det_output_buffer, det_output_size_);
    ret = aclmdlAddDatasetBuffer(det_output_dataset_, det_output_data);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Add det output dataset buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    INFO_LOG("Model initialized. Input size: %zu, Output size: %zu",
             det_input_buffer_size_, det_output_size_);
    return SUCCESS;
}

Result AscendOCR::ProcessImage(const string& image_path) {
    // 读取原始图像
    src_image_ = imread(image_path);
    if (src_image_.empty()) {
        ERROR_LOG("Failed to read image: %s", image_path.c_str());
        return FAILED;
    }

    INFO_LOG("Image loaded: %dx%d", src_image_.cols, src_image_.rows);

    // 图像预处理
    PrepareInput();
    return SUCCESS;
}

void AscendOCR::PrepareInput() {
    Mat resized_image;
    resize(src_image_, resized_image, Size(det_input_size, det_input_size));

    // 计算缩放比例 (原始图像到预处理图像)
    scale_x_ = static_cast<float>(det_input_size) / src_image_.cols;
    scale_y_ = static_cast<float>(det_input_size) / src_image_.rows;

    // 将图像数据复制到输入缓冲区
    memcpy(det_input_buffer_, resized_image.data, det_input_buffer_size_);
}

// 坐标转换函数
vector<Point2f> AscendOCR::TransformBoxToOriginal(const vector<Point2f>& box) {
    vector<Point2f> transformed_box;
    for (const auto& pt : box) {
        // 将坐标从预处理图像空间转换回原始图像空间
        transformed_box.push_back(Point2f(
                pt.x / scale_x_,
                pt.y / scale_y_
        ));
    }
    return transformed_box;
}

Result AscendOCR::DetectText() {

    INFO_LOG("Starting model inference...");

    // 执行推理
    aclError ret = aclmdlExecute(det_model_id_, det_input_dataset_, det_output_dataset_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Execute det model failed, errorCode is %d", ret);
        return FAILED;
    }

    INFO_LOG("Inference completed");

    // 获取输出数据
    aclDataBuffer* output_buffer = aclmdlGetDatasetBuffer(det_output_dataset_, 0);
    void* output_data = aclGetDataBufferAddr(output_buffer);

    // 将输出数据复制回主机
    float* det_output = new float[det_output_size_ / sizeof(float)];
    ret = aclrtMemcpy(det_output, det_output_size_, output_data, det_output_size_,
                      (run_mode_ == ACL_DEVICE) ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_HOST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Copy det output failed, errorCode is %d", ret);
        delete[] det_output;
        return FAILED;
    }

    INFO_LOG("Output data copied to host");

    // 后处理 - 检测框提取
    PostprocessDet(det_output, det_output_size_);

    delete[] det_output;
    return SUCCESS;
}

void AscendOCR::PostprocessDet(float* output_data, size_t output_size) {
    // 假设输出是概率图，大小为736x736（与输入相同）
    int height = det_input_size;
    int width = det_input_size;

    INFO_LOG("Postprocessing output. Assuming size: %dx%d", width, height);

    // 创建概率图
    Mat prob_map(height, width, CV_32FC1, output_data);

    // 二值化
    Mat bin_map;
    threshold(prob_map, bin_map, det_threshold, 1.0, THRESH_BINARY);
    bin_map.convertTo(bin_map, CV_8UC1);

    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(bin_map, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    INFO_LOG("Found %ld contours", contours.size());

    // 处理每个轮廓
    for (const auto& contour : contours) {
        // 获取最小外接矩形
        RotatedRect rect = minAreaRect(contour);
        vector<Point2f> box(4);
        rect.points(box.data());

        // 计算框得分
        float score = BoxScoreFast(prob_map, box);
        if (score < box_threshold) {
            continue;
        }

        // 解压缩框
        vector<Point2f> unclipped_box = UnclipBox(box);
        if (unclipped_box.empty()) {
            continue;
        }

        // 再次获取最小外接矩形
        RotatedRect unclipped_rect = minAreaRect(unclipped_box);
        vector<Point2f> final_box(4);
        unclipped_rect.points(final_box.data());

        // 排序点
        OrderPointsClockwise(final_box);

        // 将框坐标转换回原始图像空间
        vector<Point2f> original_box = TransformBoxToOriginal(final_box);

        // 添加到结果
        dt_boxes_.push_back(original_box);
    }

    INFO_LOG("Detected %ld text boxes", dt_boxes_.size());

    // 过滤无效框
//    FilterBoxes();
//    INFO_LOG("After filtering: %ld text boxes", dt_boxes_.size());
}

void AscendOCR::OrderPointsClockwise(vector<Point2f>& box) {
    if (box.size() != 4) return;

    // 1. 计算中心点
    Point2f center(0, 0);
    for (const auto& pt : box) {
        center += pt;
    }
    center.x /= 4;
    center.y /= 4;

    // 2. 分离左上、右上、右下、左下点
    vector<Point2f> top, bottom;
    for (const auto& pt : box) {
        if (pt.y < center.y) {
            top.push_back(pt);
        } else {
            bottom.push_back(pt);
        }
    }

    // 3. 排序左上和右上（按x坐标升序）
    if (top.size() >= 2) {
        sort(top.begin(), top.end(), [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
        });
    }

    // 4. 排序左下和右下（按x坐标降序）
    if (bottom.size() >= 2) {
        sort(bottom.begin(), bottom.end(), [](const Point2f& a, const Point2f& b) {
            return a.x > b.x;
        });
    }

    // 5. 组合点
    if (top.size() == 2 && bottom.size() == 2) {
        box[0] = top[0]; // 左上
        box[1] = top[1]; // 右上
        box[2] = bottom[0]; // 右下
        box[3] = bottom[1]; // 左下
    }
}

vector<Point2f> AscendOCR::UnclipBox(const vector<Point2f>& box) {
    // 计算多边形面积
    double area = contourArea(box);
    if (area < 1e-5) {
        return {};
    }

    // 计算距离 = 面积 * unclip_ratio / 周长
    double length = arcLength(box, true);
    double distance = area * unclip_ratio / length;

    // 关键修改：增加扩展强度
    distance *= 1.0;  // 增加扩展强度

    // 使用形态学膨胀方法进行扩展
    vector<Point> int_box;
    for (const auto& pt : box) {
        int_box.push_back(Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }

    // 创建二值图像
    Mat mask = Mat::zeros(Size(det_input_size, det_input_size), CV_8UC1);
    vector<vector<Point>> contours = {int_box};
    drawContours(mask, contours, 0, Scalar(255), FILLED);

    // 计算膨胀核大小 - 关键修改：增加核大小
    int kernel_size = static_cast<int>(distance * 3);  // 增加膨胀强度
    if (kernel_size < 1) kernel_size = 1;

    // 创建膨胀核
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));

    // 执行膨胀操作
    Mat expanded_mask;
    dilate(mask, expanded_mask, kernel);

    // 查找膨胀后的轮廓
    vector<vector<Point>> new_contours;
    findContours(expanded_mask, new_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (new_contours.empty()) {
        return {};
    }

    // 取面积最大的轮廓
    double max_area = 0;
    int max_index = 0;
    for (int i = 0; i < new_contours.size(); i++) {
        double area = contourArea(new_contours[i]);
        if (area > max_area) {
            max_area = area;
            max_index = i;
        }
    }

    // 获取最小外接矩形
    RotatedRect rect = minAreaRect(new_contours[max_index]);
    vector<Point2f> expanded_box(4);
    rect.points(expanded_box.data());

    return expanded_box;
}

float AscendOCR::BoxScoreFast(Mat& bitmap, const vector<Point2f>& box) {
    // 创建掩码
    Mat mask = Mat::zeros(bitmap.size(), CV_8UC1);
    vector<Point> int_box;
    for (const auto& pt : box) {
        int_box.push_back(Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }
    vector<vector<Point>> contours = {int_box};
    fillPoly(mask, contours, Scalar(1));

    // 计算得分 - 使用整个掩码区域
    return static_cast<float>(mean(bitmap, mask)[0]);
}

void AscendOCR::FilterBoxes() {
    // 过滤无效检测框
    vector<vector<Point2f>> filtered_boxes;

    // 进一步放宽过滤条件
    const float min_height = 5.0f;   // 降低最小高度要求
    const float max_ratio = 25.0f;   // 增加最大宽高比
    const float min_area = 25.0f;    // 降低最小面积要求

    for (auto& box : dt_boxes_) {
        // 计算框的高度和宽高比
        float height = norm(box[3] - box[0]);
        float width = norm(box[1] - box[0]);
        float ratio = width / height;

        // 计算面积
        float area = contourArea(box);

        // 添加长宽检查
        bool valid_size = width > 5.0f && height > 5.0f;

        if (height > min_height && ratio < max_ratio && area > min_area && valid_size) {
            filtered_boxes.push_back(box);
        } else {
            INFO_LOG("Filtered box: h=%.1f, w=%.1f, ratio=%.1f, area=%.1f",
                     height, width, ratio, area);
        }
    }
    dt_boxes_ = filtered_boxes;
}

void AscendOCR::VisualizeResults(const string& output_path) {
    if (dt_boxes_.empty()) {
        INFO_LOG("No text boxes detected");
        return;
    }

    Mat result_image = src_image_.clone();

    for (const auto& box : dt_boxes_) {
        // 绘制四边形
        vector<Point> int_box;
        for (const auto& pt : box) {
            int_box.push_back(Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }

        // 绘制边界框
        polylines(result_image, int_box, true, Scalar(0, 255, 0), 2);

        // 绘制顶点
        for (const auto& pt : int_box) {
            circle(result_image, pt, 3, Scalar(0, 0, 255), -1);
        }
    }

    // 保存结果
    imwrite(output_path, result_image);
    INFO_LOG("Visualization saved to: %s", output_path.c_str());
}

void AscendOCR::ReleaseResource() {
    // 释放检测模型资源
    if (det_input_dataset_) {
        aclmdlDestroyDataset(det_input_dataset_);
        det_input_dataset_ = nullptr;
    }
    if (det_output_dataset_) {
        aclmdlDestroyDataset(det_output_dataset_);
        det_output_dataset_ = nullptr;
    }
    if (det_model_desc_) {
        aclmdlDestroyDesc(det_model_desc_);
        det_model_desc_ = nullptr;
    }
    if (det_model_id_) {
        aclmdlUnload(det_model_id_);
        det_model_id_ = 0;
    }

    // 释放ACL资源
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }

    aclrtResetDevice(0);
    aclFinalize();
}

int main(int argc, char* argv[]) {

    const char* det_model_path = "/home/HwHiAiUser/ocr/models/ch_PP-OCRv4_det_infer.om";
    string image_path = "/home/HwHiAiUser/ocr/pic/ocr_water.png";
    string output_path = "/home/HwHiAiUser/ocr/out/ocr_water_out.png";

    AscendOCR ocr(det_model_path);

    // 初始化OCR系统
    Result ret = ocr.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("OCR initialization failed");
        return FAILED;
    }

    // 处理图像
    ret = ocr.ProcessImage(image_path);
    if (ret != SUCCESS) {
        ERROR_LOG("Image processing failed");
        return FAILED;
    }

    // 文本检测
    ret = ocr.DetectText();
    if (ret != SUCCESS) {
        ERROR_LOG("Text detection failed");
        return FAILED;
    }

    // 获取检测框
    vector<vector<Point2f>> boxes = ocr.GetBoxes();
    INFO_LOG("Detected %ld text boxes", boxes.size());

    // 可视化结果
    ocr.VisualizeResults(output_path);

    return SUCCESS;
}