#ifndef SAMPLERESNETQUICKSTART_ASCENDOCR_H
#define SAMPLERESNETQUICKSTART_ASCENDOCR_H

#include "acl/acl.h"
#include "Ascend_rec.h"
#include <opencv2/opencv.hpp>

namespace {
    const int det_input_size = 736;
    const float det_threshold = 0.3f;
    const float box_threshold = 0.5f;
    const float unclip_ratio = 1.2f;
}

class AscendOCR {
public:
    AscendOCR(const char* det_model_path, const char* rec_model_path, const char* rec_keys_path);
    ~AscendOCR();
    Result Init();
    Result ProcessImage(const string& image_path);
    Result DetectText();
    vector<vector<Point2f>> GetBoxes() { return dt_boxes_; }
    void VisualizeResults(const string& output_path);

    // 文本识别函数
    Result RecognizeText();
    vector<string> GetRecTexts() { return rec_texts_; }

    size_t rec_input_buffer_size_;
    size_t rec_output_size_;
    int rec_seq_length_;
    // 存储序列长度
    int rec_num_classes_;

    //检查抠出来的图是不是水尺部分
    bool IsWaterScaleCroppedImage(const cv::Mat& image);

    //裁剪抠出来的图，提高精度
    cv::Mat CropBorders(const cv::Mat& image, float top_bottom_ratio, float left_right_ratio, bool enhance);

    //增强对比度，提高精度
    void EnhanceWaterScaleImage(cv::Mat& image);

    //检测红色箭头底部中心点
    void DetectRedArrows(std::string image_path);

    //简单实现水位线检测，实际换为你们的
    int DetectWaterline(const cv::Mat& image);

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

    // 识别模型相关变量
    const char* rec_model_path_;
    const char* rec_keys_path_;
    uint32_t rec_model_id_;
    aclmdlDesc* rec_model_desc_;
    aclmdlDataset* rec_input_dataset_;
    aclmdlDataset* rec_output_dataset_;
    void* rec_input_buffer_;

    // 识别相关变量
    vector<string> rec_texts_;  // 存储识别结果

    // 识别模型初始化函数
    Result InitRecModel();

    std::vector<std::string> char_list_;

    std::vector<int> ignored_tokens_;     // 需要忽略的token索引

    AscendTextRecognizer* text_recognizer_;

    struct DetectedLine {
        cv::Rect rect;
        int position; // y坐标位置
    };

    std::vector<DetectedLine> DetectHorizontalLines(cv::Mat& image);

};


#endif //SAMPLERESNETQUICKSTART_ASCENDOCR_H
