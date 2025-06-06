#include "AscendOCR.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)

AscendOCR::AscendOCR(const char* det_model_path, const char* rec_model_path, const char* rec_keys_path)
        : det_model_path_(det_model_path), rec_model_path_(rec_model_path),rec_keys_path_(rec_keys_path),
          context_(nullptr), stream_(nullptr),
          det_model_id_(0), det_model_desc_(nullptr),
          det_input_dataset_(nullptr), det_output_dataset_(nullptr),
          det_input_buffer_(nullptr), det_output_size_(0)
{
    text_recognizer_ = new AscendTextRecognizer(rec_model_path, rec_keys_path);
}

AscendOCR::~AscendOCR() {
    ReleaseResource();
    if (text_recognizer_) {
        delete text_recognizer_;
        text_recognizer_ = nullptr;
    }
}

Result AscendOCR::InitRecModel() {
    // 加载识别模型
    aclError ret = aclmdlLoadFromFile(rec_model_path_, &rec_model_id_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Load rec model failed, errorCode is %d", ret);
        return FAILED;
    }

    rec_model_desc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(rec_model_desc_, rec_model_id_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Get rec model desc failed, errorCode is %d", ret);
        return FAILED;
    }

    // 创建识别模型输入输出数据集
    rec_input_dataset_ = aclmdlCreateDataset();
    rec_output_dataset_ = aclmdlCreateDataset();

    // 获取输入缓冲区
    size_t rec_input_index = 0;
    rec_input_buffer_size_ = aclmdlGetInputSizeByIndex(rec_model_desc_, rec_input_index);

    // 验证输入大小是否符合预期
    const size_t expected_input_size = 1 * 3 * 48 * 320 * sizeof(float); // NCHW格式
    if (rec_input_buffer_size_ != expected_input_size) {
        WARN_LOG("Rec model input size: %zu, expected: %zu. Adjusting...",
                 rec_input_buffer_size_, expected_input_size);
        rec_input_buffer_size_ = expected_input_size;
    }

    // 分配输入缓冲区
    ret = aclrtMalloc(&rec_input_buffer_, rec_input_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Malloc rec input buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    aclDataBuffer* rec_input_data = aclCreateDataBuffer(rec_input_buffer_, rec_input_buffer_size_);
    ret = aclmdlAddDatasetBuffer(rec_input_dataset_, rec_input_data);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Add rec input dataset buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    // 获取输出缓冲区
    size_t rec_output_index = 0;
    rec_output_size_ = aclmdlGetOutputSizeByIndex(rec_model_desc_, rec_output_index);
    void* rec_output_buffer;
    ret = aclrtMalloc(&rec_output_buffer, rec_output_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Malloc rec output buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    aclDataBuffer* rec_output_data = aclCreateDataBuffer(rec_output_buffer, rec_output_size_);
    ret = aclmdlAddDatasetBuffer(rec_output_dataset_, rec_output_data);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Add rec output dataset buffer failed, errorCode is %d", ret);
        return FAILED;
    }

    // 获取输出维度信息
    aclmdlIODims output_dims;
    ret = aclmdlGetOutputDims(rec_model_desc_, rec_output_index, &output_dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Get output dims failed, error: %d", ret);
        return FAILED;
    }

    // 计算实际序列长度和类别数
    rec_seq_length_ = output_dims.dims[1];
    rec_num_classes_ = output_dims.dims[2];

    // 确保字符表大小匹配模型输出
    if (rec_num_classes_ != char_list_.size()) {
        WARN_LOG("Class count mismatch: model=%d, char_list=%zu. Adjusting to model size.",
                 rec_num_classes_, char_list_.size());

        // 调整字符表大小以匹配模型
        if (rec_num_classes_ > char_list_.size()) {
            // 添加未知字符占位符
            size_t diff = rec_num_classes_ - char_list_.size();
            for (size_t i = 0; i < diff; i++) {
                char_list_.push_back("<UNK_" + to_string(i) + ">");
            }
        } else {
            // 截断字符表
            char_list_.resize(rec_num_classes_);
        }
    }

    // 设置忽略的token（空白符）
    ignored_tokens_ = {0};

    INFO_LOG("Rec model initialized:");
    INFO_LOG("  Input size: %zu", rec_input_buffer_size_);
    INFO_LOG("  Output size: %zu", rec_output_size_);
    INFO_LOG("  Sequence length: %d", rec_seq_length_);
    INFO_LOG("  Number of classes: %d", rec_num_classes_);
    INFO_LOG("  Character list size: %zu", char_list_.size());

    return SUCCESS;
}

Result AscendOCR::Init() {
    // 初始化ACL环境，用到2个模型，但只初始化一次
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

    // 2. 创建文本识别器
    text_recognizer_ = new AscendTextRecognizer(rec_model_path_, rec_keys_path_);

    if (text_recognizer_->Init(context_, stream_) != SUCCESS) {
        ERROR_LOG("Text recognizer initialization failed");
        return FAILED;
    }

//    ret = aclrtCreateContext(&context_, 0);
//    if (ret != ACL_SUCCESS) {
//        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
//        return FAILED;
//    }
//
//    ret = aclrtCreateStream(&stream_);
//    if (ret != ACL_SUCCESS) {
//        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
//        return FAILED;
//    }

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

    // 初始化识别模型
    ret = InitRecModel();
    if (ret != SUCCESS) {
        ERROR_LOG("Rec model initialization failed");
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

bool AscendOCR::IsWaterScaleCroppedImage(const cv::Mat& image) {
    if (image.empty()) return false;

    // 1. 颜色分析（红色区域占比）
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red_mask;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(15, 255, 255), red_mask);
    cv::Mat red_mask2;
    cv::inRange(hsv, cv::Scalar(160, 50, 50), cv::Scalar(180, 255, 255), red_mask2);
    cv::bitwise_or(red_mask, red_mask2, red_mask);

    double red_ratio = cv::countNonZero(red_mask) / (double)image.total();

    // 2. 背景分析（白色区域占比）
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat white_mask;
    cv::threshold(gray, white_mask, 200, 255, cv::THRESH_BINARY);
    double white_ratio = cv::countNonZero(white_mask) / (double)image.total();

    // 3. 判定为水尺图像的条件
    bool is_water = (red_ratio > 0.05) && (white_ratio > 0.3);

    INFO_LOG("Cropped image: red=%.3f, white=%.3f -> %s",
             red_ratio, white_ratio, is_water ? "WaterScale" : "Normal");

    return is_water;
}

cv::Mat AscendOCR::CropBorders(const cv::Mat& image, float top_bottom_ratio, float left_right_ratio, bool enhance) {
    if (image.empty()) return image;

    // 计算裁剪尺寸
    int top = static_cast<int>(image.rows * top_bottom_ratio);
    int bottom = static_cast<int>(image.rows * top_bottom_ratio);
    int left = static_cast<int>(image.cols * left_right_ratio);
    int right = static_cast<int>(image.cols * left_right_ratio);

    // 确保裁剪后还有有效区域
    if (top + bottom >= image.rows) {
        top = static_cast<int>(image.rows * 0.05);
        bottom = static_cast<int>(image.rows * 0.05);
        WARN_LOG("Top+bottom crop too large, reduced to 5%% each");
    }

    if (left + right >= image.cols) {
        left = static_cast<int>(image.cols * 0.05);
        right = static_cast<int>(image.cols * 0.05);
        WARN_LOG("Left+right crop too large, reduced to 5%% each");
    }

    // 创建裁剪区域
    cv::Rect roi(left, top, image.cols - left - right, image.rows - top - bottom);
    if (roi.width <= 0 || roi.height <= 0) {
        WARN_LOG("Invalid crop region, skipping border crop");
        return image;
    }

    cv::Mat result = image(roi).clone();

    // 应用图像增强（如果需要）
    if (enhance) {
        EnhanceWaterScaleImage(result);
    }

    return result;
}

void AscendOCR::EnhanceWaterScaleImage(cv::Mat& image) {
    if (image.empty()) return;

    // 1. 提取红色区域并增强对比度
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 红色范围定义（包括浅红和深红）
    cv::Mat red_mask1, red_mask2;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), red_mask1);
    cv::inRange(hsv, cv::Scalar(160, 120, 70), cv::Scalar(180, 255, 255), red_mask2);
    cv::Mat red_mask;
    cv::bitwise_or(red_mask1, red_mask2, red_mask);

    // 增强红色区域
    cv::Mat enhanced = image.clone();
    for (int c = 0; c < 3; c++) {
        if (c != 2) { // 非红色通道
            enhanced.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) {
                if (red_mask.at<uchar>(position[0], position[1])) {
                    pixel[c] = cv::saturate_cast<uchar>(pixel[c] * 0.7); // 降低其他通道强度
                }
            });
        } else { // 红色通道
            enhanced.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) {
                if (red_mask.at<uchar>(position[0], position[1])) {
                    pixel[c] = cv::saturate_cast<uchar>(pixel[c] * 1.3 + 40); // 增强红色
                }
            });
        }
    }

    // 2. 增强小数点（暗点）
    cv::Mat gray;
    cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);

    // 查找暗点（小数点）
    cv::Mat dark_mask;
    cv::threshold(gray, dark_mask, 80, 255, cv::THRESH_BINARY_INV);

    // 形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_OPEN, kernel);

    // 增强暗点区域
    enhanced.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) {
        if (dark_mask.at<uchar>(position[0], position[1])) {
            for (int c = 0; c < 3; c++) {
                pixel[c] = cv::saturate_cast<uchar>(pixel[c] * 0.7); // 加深暗点
            }
        }
    });

    // 3. 整体对比度增强
    cv::Mat lab_image;
    cv::cvtColor(enhanced, lab_image, cv::COLOR_BGR2Lab);

    // 分离L通道并增强
    std::vector<cv::Mat> lab_planes;
    cv::split(lab_image, lab_planes);

    // CLAHE（对比度受限自适应直方图均衡化）
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(lab_planes[0], lab_planes[0]);

    // 合并通道并转换回BGR
    cv::merge(lab_planes, lab_image);
    cv::cvtColor(lab_image, image, cv::COLOR_Lab2BGR);

    // 4. 锐化处理（增强边缘）
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(0, 0), 2.0);
    cv::addWeighted(image, 1.5, blurred, -0.5, 0, image);

    // 5. 测试：保存增强后的图像用于调试
//    static int enhance_count = 0;
//    if (enhance_count < 10) { // 限制保存数量
//        cv::imwrite("/home/HwHiAiUser/ocr/debug/enhanced_" +
//                    to_string(enhance_count++) + ".png", image);
//    }
}

Result AscendOCR::RecognizeText() {
    if (dt_boxes_.empty()) {
        INFO_LOG("No text boxes to recognize");
        return SUCCESS;
    }

    rec_texts_.clear();
    vector<Mat> cropped_images;

    // 为每个检测框准备裁剪图像
    for (int idx = 0; idx < dt_boxes_.size(); idx++) {
        const auto& box = dt_boxes_[idx];

        // 1. 计算最小外接旋转矩形
        RotatedRect rotated_rect = minAreaRect(box);

        // 2. 获取旋转矩形的角度和尺寸
        float angle = rotated_rect.angle;
        Size rect_size = rotated_rect.size;

        // 调整角度（OpenCV的旋转矩形角度范围特殊）
        if (angle < -45) {
            angle += 90;
            swap(rect_size.width, rect_size.height);
        }

        // 3. 获取旋转矩阵
        Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, angle, 1.0);

        // 4. 执行旋转矫正
        Mat rotated_image;
        warpAffine(src_image_, rotated_image, rotation_matrix, src_image_.size(), INTER_CUBIC);

        // 5. 裁剪矫正后的矩形区域
        Rect rect_area(rotated_rect.center.x - rect_size.width/2,
                       rotated_rect.center.y - rect_size.height/2,
                       rect_size.width, rect_size.height);

        // 边界检查
        rect_area.x = max(0, rect_area.x);
        rect_area.y = max(0, rect_area.y);
        rect_area.width = min(rotated_image.cols - rect_area.x, rect_size.width);
        rect_area.height = min(rotated_image.rows - rect_area.y, rect_size.height);

        if (rect_area.width <= 0 || rect_area.height <= 0) {
            cropped_images.push_back(Mat());
            continue;
        }

        Mat cropped_img = rotated_image(rect_area).clone();
        if (IsWaterScaleCroppedImage(cropped_img)) {
            // 四边裁剪
            cropped_img = CropBorders(cropped_img, 0.1,0.15,true);

            // 测试：保存裁剪后的图像用于调试
//            static int crop_count = 0;
//            cv::imwrite("/home/HwHiAiUser/ocr/debug/cropped_" +
//                        to_string(crop_count++) + ".png", cropped_img);
        }
        cropped_images.push_back(cropped_img);
    }

    // 使用文本识别器进行批量识别
    if (text_recognizer_->RecognizeText(cropped_images, rec_texts_) != SUCCESS) {
        ERROR_LOG("Text recognition failed");
        return FAILED;
    }

    return SUCCESS;
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
    // 大小为736x736（与输入相同）
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
        // 跳过小轮廓
        if (contour.size() < 5) {
            continue;
        }

        // 获取最小外接矩形
        RotatedRect rect = minAreaRect(contour);
        vector<Point2f> box(4);
        rect.points(box.data());

        // 计算框得分
        float score = BoxScoreFast(prob_map, box);
        if (score < box_threshold) {
            continue;
        }

        const float min_box_size = 10.0f;  // 最小框尺寸
        const float max_aspect_ratio = 50.0f; // 最大宽高比

        // 计算框的尺寸
        float width = norm(box[1] - box[0]);
        float height = norm(box[3] - box[0]);

        // 过滤无效框
        if (width < min_box_size || height < min_box_size ||
            width/height > max_aspect_ratio || height/width > max_aspect_ratio) {
            continue;
        }

        // 解压缩框
        vector<Point2f> unclipped_box = UnclipBox(box);
        if (unclipped_box.empty()) {
            continue;
        }

        bool valid = true;
        for (const auto& pt : unclipped_box) {
            if (isnan(pt.x) || isnan(pt.y)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            WARN_LOG("Invalid points in unclipped box");
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

        OrderPointsClockwise(final_box);

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

    // 1. 检查点坐标是否有效
    for (const auto& pt : box) {
        if (isnan(pt.x) || isnan(pt.y)) {
            ERROR_LOG("Invalid point in box: (%.1f, %.1f)", pt.x, pt.y);
            return;
        }
    }

    // 2. 计算中心点
    Point2f center(0, 0);
    for (const auto& pt : box) {
        center += pt;
    }
    center.x /= 4;
    center.y /= 4;

    // 3. 分离左上、右上、右下、左下点
    vector<Point2f> top, bottom;
    for (const auto& pt : box) {
        if (pt.y < center.y) {
            top.push_back(pt);
        } else {
            bottom.push_back(pt);
        }
    }

    // 4. 处理特殊情况 - 确保至少有2个点
    if (top.size() < 2 || bottom.size() < 2) {
        // 直接使用最小外接矩形并排序
        RotatedRect rect = minAreaRect(box);
        rect.points(box.data());

        // 按顺时针排序：左上->右上->右下->左下
        sort(box.begin(), box.end(), [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
        });
        return;
    }

    // 5. 排序左上和右上（按x坐标升序）
    sort(top.begin(), top.end(), [](const Point2f& a, const Point2f& b) {
        return a.x < b.x;
    });

    // 6. 排序左下和右下（按x坐标降序）
    sort(bottom.begin(), bottom.end(), [](const Point2f& a, const Point2f& b) {
        return a.x > b.x;
    });

    // 7. 组合点 - 确保正确的顺序
    box[0] = top[0]; // 左上
    box[1] = top[1]; // 右上
    box[2] = bottom[0]; // 右下
    box[3] = bottom[1]; // 左下
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
        if (isnan(pt.x) || isnan(pt.y)) {
            ERROR_LOG("Invalid point in box: (%.1f, %.1f)", pt.x, pt.y);
            return {};
        }
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

    for (int i = 0; i < dt_boxes_.size(); i++) {
        const auto& box = dt_boxes_[i];
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

        // 绘制识别文本（在框的上方）
        if (i < rec_texts_.size()) {
            string text = rec_texts_[i];
            putText(result_image, text, int_box[0], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }
    }

    // 保存结果
//    imwrite(output_path, result_image);
//    INFO_LOG("Visualization saved to: %s", output_path.c_str());
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
    const char* rec_model_path = "/home/HwHiAiUser/ocr/models/ch_PP-OCRv4_rec_infer.om";
    const char* rec_keys_path = "/home/HwHiAiUser/ocr/models/ppocr_keys_v1.txt";
    string image_path = "/home/HwHiAiUser/ocr/pic/ocr_water.png";
    string output_path = "/home/HwHiAiUser/ocr/out/ocr_water_out.png";

    AscendOCR ocr(det_model_path, rec_model_path,rec_keys_path);

    // 初始化OCR系统
    Result ret = ocr.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("OCR initialization failed");
        return FAILED;
    }

    INFO_LOG("Recognition model info:");
    INFO_LOG("  Input size: %zu", ocr.rec_input_buffer_size_);
    INFO_LOG("  Output size: %zu", ocr.rec_output_size_);
    INFO_LOG("  Sequence length: %d", ocr.rec_seq_length_);
    INFO_LOG("  Number of classes: %d", ocr.rec_num_classes_);

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

    // 文本识别
    ret = ocr.RecognizeText();
    if (ret != SUCCESS) {
        ERROR_LOG("Text recognition failed");
        return FAILED;
    }

    // 获取识别结果
    vector<string> rec_texts = ocr.GetRecTexts();
    INFO_LOG("Recognized %ld text results", rec_texts.size());

    // 打印识别结果
    for (int i = 0; i < rec_texts.size(); i++) {
        INFO_LOG("Box %d: %s", i, rec_texts[i].c_str());
    }

    // 可视化结果（包含检测框和识别文本）
    ocr.VisualizeResults(output_path);

    return SUCCESS;
}