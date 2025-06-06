#ifndef SAMPLERESNETQUICKSTART_ASCEND_REC_H
#define SAMPLERESNETQUICKSTART_ASCEND_REC_H

#include <opencv2/opencv.hpp>
#include "acl/acl.h"

using namespace cv;
using namespace std;


typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class AscendTextRecognizer {
public:
    AscendTextRecognizer(const char* rec_model_path, const char* rec_keys_path);
    ~AscendTextRecognizer();
//    Result Init();
    Result Init(aclrtContext context, aclrtStream stream);
    Result RecognizeText(const vector<Mat>& cropped_images, vector<string>& rec_texts);
    void ReleaseResource();

private:
    // 模型相关变量
    aclrtContext context_;
    aclrtStream stream_;
    aclrtRunMode run_mode_;

    // 识别模型相关变量
    const char* rec_model_path_;
    uint32_t rec_model_id_;
    aclmdlDesc* rec_model_desc_;
    aclmdlDataset* rec_input_dataset_;
    aclmdlDataset* rec_output_dataset_;
    void* rec_input_buffer_;
    size_t rec_input_buffer_size_;
    size_t rec_output_size_;

    // 识别相关变量
    vector<string> char_list_;  // 字符映射表
    vector<int> ignored_tokens_; // 需要忽略的token索引

    // 模型参数
    int rec_seq_length_;
    int rec_num_classes_;

    // 字符映射表加载
    void LoadCharList(const char* rec_keys_path);

    // 预处理函数
    void PreprocessCroppedImage(const Mat& cropped_img, Mat& preprocessed_img);

    // CTC解码核心函数
    std::string CTCDecode(float* output_data, int output_size);

    // 应用Softmax
    void ApplySoftmax(float* data, int length);
};
#endif //SAMPLERESNETQUICKSTART_ASCEND_REC_H
