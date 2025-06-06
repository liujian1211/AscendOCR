#include "Ascend_rec.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)

AscendTextRecognizer::AscendTextRecognizer(const char* rec_model_path, const char* rec_keys_path)
        : rec_model_path_(rec_model_path),
          context_(nullptr), stream_(nullptr),
          rec_model_id_(0), rec_model_desc_(nullptr),
          rec_input_dataset_(nullptr), rec_output_dataset_(nullptr),
          rec_input_buffer_(nullptr), rec_output_size_(0),
          rec_seq_length_(0), rec_num_classes_(0) {

    LoadCharList(rec_keys_path);
}

AscendTextRecognizer::~AscendTextRecognizer() {
    ReleaseResource();
}

void AscendTextRecognizer::LoadCharList(const char* rec_keys_path) {
    std::ifstream keys_file(rec_keys_path);
    char_list_.clear();
    std::string line;

    // 1. 必须确保第一个字符是空白符（用于CTC解码）
    char_list_.push_back(" ");

    // 2. 从文件加载字符表
    while (std::getline(keys_file, line)) {
        // 移除行尾的换行符
        if (!line.empty() && line[line.size()-1] == '\n') {
            line.erase(line.size()-1);
        }
        if (!line.empty() && line[line.size()-1] == '\r') {
            line.erase(line.size()-1);
        }

        // 跳过空行
        if (!line.empty()) {
            char_list_.push_back(line);
        }
    }
    keys_file.close();

    // 3. 记录加载的字符信息（用于调试）
    INFO_LOG("Loaded %zu characters from %s", char_list_.size(), rec_keys_path);
    INFO_LOG("First character: [%s]", char_list_[0].c_str());
    INFO_LOG("Last character: [%s]", char_list_.back().c_str());
    if (char_list_.size() > 100) {
        INFO_LOG("Character at index 100: [%s]", char_list_[100].c_str());
    }

    // 4. 设置需要忽略的token（空白符，索引0）
    ignored_tokens_ = {0};
}

void AscendTextRecognizer::ApplySoftmax(float* data, int length) {
    float max_val = *std::max_element(data, data + length);
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    if (sum > 0) {
        for (int i = 0; i < length; i++) {
            data[i] /= sum;
        }
    }
}

std::string AscendTextRecognizer::CTCDecode(float* output_data, int output_size) {
    const int seq_length = rec_seq_length_;
    const int num_classes = rec_num_classes_;

    // 验证输出大小是否足够
    if (static_cast<size_t>(output_size) < seq_length * num_classes) {
        ERROR_LOG("Output size too small: %d < %d", output_size, seq_length * num_classes);
        return "";
    }

    // 1. 获取每个时间步的最大概率索引
    std::vector<int> preds_idx(seq_length);
    for (int t = 0; t < seq_length; t++) {
        float* probs = output_data + t * num_classes;
        preds_idx[t] = std::max_element(probs, probs + num_classes) - probs;
    }

    // 2. 去除连续重复字符（标准CTC解码步骤）
    std::vector<int> deduplicated;
    int prev_idx = -1;
    for (int t = 0; t < seq_length; t++) {
        int current_idx = preds_idx[t];
        if (current_idx != prev_idx) {
            deduplicated.push_back(current_idx);
        }
        prev_idx = current_idx;
    }

    // 3. 构建最终文本（跳过空白符）
    std::string text;
    for (int idx : deduplicated) {
        // 跳过空白符（索引0）
        if (idx == 0) continue;

        // 检查索引是否在有效范围内
        if (idx < static_cast<int>(char_list_.size())) {
            text += char_list_[idx];
        } else {
            // 对于无效索引，使用占位符
            WARN_LOG("Invalid character index: %d ", idx);
            text += "<?>";
        }
    }

    // 4. 调试输出：打印每个时间步的预测结果
    if (text.empty()) {
        INFO_LOG("No valid characters detected. Time step predictions:");

        // 只打印前20个时间步（避免日志过长）
        int max_steps_to_print = std::min(20, seq_length);
        for (int t = 0; t < max_steps_to_print; t++) {
            float* probs = output_data + t * num_classes;
            int max_idx = std::max_element(probs, probs + num_classes) - probs;
            float max_prob = probs[max_idx];

            string char_str = (max_idx < char_list_.size()) ?
                              char_list_[max_idx] : "UNKNOWN";

//            INFO_LOG("  Step %2d: [%s] (idx=%4d) prob=%.4f",
//                     t, char_str.c_str(), max_idx, max_prob);
        }
    }

    return text;
}

Result AscendTextRecognizer::Init(aclrtContext context, aclrtStream stream) {
    // 初始化ACL环境
//    aclError ret = aclInit(nullptr);
//    if (ret != ACL_SUCCESS) {
//        ERROR_LOG("aclInit failed, errorCode is %d", ret);
//        return FAILED;
//    }

    context_ = context;
    stream_ = stream;

    aclError ret = aclrtSetDevice(0); // 使用设备0
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

    // 加载识别模型
    ret = aclmdlLoadFromFile(rec_model_path_, &rec_model_id_);
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
    const size_t expected_input_size = 1 * 48 * 320*3;
    if (rec_input_buffer_size_ < expected_input_size) {
        ERROR_LOG("Input buffer too small: %zu < %zu", rec_input_buffer_size_, expected_input_size);
        return FAILED;
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

    // 关键修复：确保字符表大小匹配模型输出
    if (rec_num_classes_ != char_list_.size()) {
        WARN_LOG("Class count mismatch: model=%d, char_list=%zu. Adjusting to model size.",
                 rec_num_classes_, char_list_.size());

        // 检查是否缺少空白符
        if (char_list_.empty() || char_list_[0] != " ") {
            WARN_LOG("Blank character missing! Adding blank character at index 0.");
            char_list_.insert(char_list_.begin(), " ");
        }

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
            // 再次确保空白符在索引0位置
            if (char_list_[0] != " ") {
                WARN_LOG("Blank character not at index 0 after resize!");
                char_list_[0] = " ";
            }
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

void AscendTextRecognizer::PreprocessCroppedImage(const Mat& cropped_img, Mat& preprocessed_img) {
    // 确保输入图像有效
    if (cropped_img.empty()) {
        ERROR_LOG("Cropped image is empty in PreprocessCroppedImage");
        return;
    }

    // 1. 转换为灰度图（如果是彩色图）
    Mat gray;
    if (cropped_img.channels() == 3) {
        cvtColor(cropped_img, gray, COLOR_BGR2GRAY);
    } else {
        gray = cropped_img.clone();
    }

    // 2. 调整大小 - 保持宽高比
    const int rec_height = 48;  // 模型期望高度48
    float wh_ratio = static_cast<float>(gray.cols) / gray.rows;
    int rec_width = min(static_cast<int>(rec_height * wh_ratio), 320);

    // 3. 调整图像大小
    Mat resized;
    resize(gray, resized, Size(rec_width, rec_height), 0, 0, INTER_LINEAR);

    // 4. 创建背景图像 (尺寸改为48x320)
    Mat norm_img(rec_height, 320, CV_8UC1, Scalar(255)); // 白色背景
    resized.copyTo(norm_img(Rect(0, 0, rec_width, rec_height)));

    // 转换为3通道 (直接创建3通道矩阵)
    Mat color_img;
    cvtColor(norm_img, color_img, COLOR_GRAY2BGR);
    cvtColor(color_img, color_img, COLOR_BGR2RGB);

    preprocessed_img = color_img.clone();

    // 8. 保存预处理后的图像用于调试
//    static int count = 0;
//    string debug_path = "/home/HwHiAiUser/ocr/debug/preprocessed_" + to_string(count) + ".png";
//    imwrite(debug_path, color_img);  // 直接保存uint8图像
//    INFO_LOG("Saved preprocessed image to: %s", debug_path.c_str());

//    if (!preprocessed_img.empty()) {
//        double minVal, maxVal;
//        minMaxLoc(preprocessed_img, &minVal, &maxVal);
//        INFO_LOG("Preprocessed data range: min=%.4f, max=%.4f", minVal, maxVal);
//    }

    // 9. 保存输入数据用于数值检查
//    char input_path[256];
//    snprintf(input_path, sizeof(input_path), "/home/HwHiAiUser/ocr/debug/input_%d.bin", count);
//    FILE* fp = fopen(input_path, "wb");
//    if (fp) {
//        fwrite(preprocessed_img.data, 1, preprocessed_img.total() * preprocessed_img.elemSize(), fp);
//        fclose(fp);
//        INFO_LOG("Saved input data to: %s (HWC format, size=%zu)",
//                 input_path, preprocessed_img.total() * preprocessed_img.elemSize());
//    }

//    count++;
}

Result AscendTextRecognizer::RecognizeText(const vector<Mat>& cropped_images, vector<string>& rec_texts) {
    if (cropped_images.empty()) {
        INFO_LOG("No cropped images to recognize");
        return SUCCESS;
    }

    rec_texts.clear();
    aclError ret;

    // 遍历所有裁剪的图像
    for (int idx = 0; idx < cropped_images.size(); idx++) {
        const Mat& cropped_img = cropped_images[idx];
        INFO_LOG("Processing cropped image %d/%d", idx+1, cropped_images.size());

        // 1. 预处理图像
        Mat preprocessed_img;
        PreprocessCroppedImage(cropped_img, preprocessed_img);

        // 检查预处理结果
        const int expected_pixels = 48 * 320;  // 期望的像素数量
        if (preprocessed_img.empty() ||
            preprocessed_img.total() != expected_pixels ||
            preprocessed_img.channels() != 3) {
            ERROR_LOG("Preprocessing failed for image %d: pixels=%zu (expected %d), channels=%d (expected 3)",
                      idx, preprocessed_img.total(), expected_pixels, preprocessed_img.channels());
            rec_texts.push_back("");
            continue;
        }

        // 2. 准备输入数据 (CHW格式)
        // 验证输入缓冲区大小
        size_t expected_input_size = 48 * 320 * 3;
        if (rec_input_buffer_size_ < expected_input_size) {
            ERROR_LOG("Input buffer size mismatch: %zu < %zu", rec_input_buffer_size_, expected_input_size);
            rec_texts.push_back("");
            continue;
        }

        // 3. 将数据复制到设备内存
        size_t actual_input_size = preprocessed_img.total() * preprocessed_img.elemSize();
        ret = aclrtMemcpy(rec_input_buffer_, rec_input_buffer_size_,
                          preprocessed_img.data, actual_input_size,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Copy input to device failed for image %d, error: %d", idx, ret);
            rec_texts.push_back("");
            continue;
        }

        // 4. 执行推理
        ret = aclmdlExecute(rec_model_id_, rec_input_dataset_, rec_output_dataset_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Execute rec model failed for image %d, error: %d", idx, ret);
            rec_texts.push_back("");
            continue;
        }

        // 5. 获取输出数据
        aclDataBuffer* output_db = aclmdlGetDatasetBuffer(rec_output_dataset_, 0);
        void* output_ptr = aclGetDataBufferAddr(output_db);
        size_t output_bytes = aclGetDataBufferSize(output_db);

        // 6. 计算预期的输出元素数量
        const size_t expected_output_elements = rec_seq_length_ * rec_num_classes_;
        const size_t expected_output_bytes = expected_output_elements * sizeof(float);

        // 7. 验证输出大小
        if (output_bytes < expected_output_bytes) {
            ERROR_LOG("Output buffer too small for image %d: %zu < %zu",
                      idx, output_bytes, expected_output_bytes);
            rec_texts.push_back("");
            continue;
        }

        // 8. 复制输出到主机内存
        vector<float> host_output(expected_output_elements);
        ret = aclrtMemcpy(host_output.data(), expected_output_bytes,
                          output_ptr, expected_output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Copy rec output failed for image %d, error: %d", idx, ret);
            rec_texts.push_back("");
            continue;
        }

        // 9. 保存输出数据用于调试
//        char output_path[256];
//        snprintf(output_path, sizeof(output_path), "/home/HwHiAiUser/ocr/debug/output_%d.bin", idx);
//        FILE* output_fp = fopen(output_path, "wb");
//        if (output_fp) {
//            fwrite(host_output.data(), 1, expected_output_bytes, output_fp);
//            fclose(output_fp);
//            INFO_LOG("Saved output data to: %s", output_path);
//        }

        // 10. 应用Softmax到每个时间步的输出
        for (int t = 0; t < rec_seq_length_; t++) {
            float* timestep_output = host_output.data() + t * rec_num_classes_;
            ApplySoftmax(timestep_output, rec_num_classes_);
        }

        // 11. 添加详细的输出统计
        vector<float> top_probs(10, 0.0f);
        vector<int> top_indices(10, 0);

        for (int t = 0; t < rec_seq_length_; t++) {
            float* timestep_output = host_output.data() + t * rec_num_classes_;

            // 找出每个时间步的前10个概率
            vector<pair<float, int>> probs_with_idx;
            for (int c = 0; c < rec_num_classes_; c++) {
                probs_with_idx.push_back({timestep_output[c], c});
            }

            partial_sort(probs_with_idx.begin(), probs_with_idx.begin() + 10, probs_with_idx.end(),
                         [](const pair<float, int>& a, const pair<float, int>& b) {
                             return a.first > b.first;
                         });

            // 打印前5个时间步的top5结果
//            if (t < 5) {
//                INFO_LOG("Time step %d top predictions:", t);
//                for (int i = 0; i < 5; i++) {
//                    int idx = probs_with_idx[i].second;
//                    float prob = probs_with_idx[i].first;
//                    string char_str = (idx < char_list_.size()) ? char_list_[idx] : "UNKNOWN";
//                    INFO_LOG("  %d: [%s] (idx=%d) prob=%.4f", i+1, char_str.c_str(), idx, prob);
//                }
//            }

            // 更新全局top概率
            for (int i = 0; i < 10; i++) {
                if (probs_with_idx[i].first > top_probs[i]) {
                    top_probs[i] = probs_with_idx[i].first;
                    top_indices[i] = probs_with_idx[i].second;
                }
            }
        }

        // 打印全局top概率
//        INFO_LOG("Global top predictions:");
//        for (int i = 0; i < 10; i++) {
//            string char_str = (top_indices[i] < char_list_.size()) ? char_list_[top_indices[i]] : "UNKNOWN";
//            INFO_LOG("  %d: [%s] (idx=%d) prob=%.4f", i+1, char_str.c_str(), top_indices[i], top_probs[i]);
//        }

        // 12. CTC解码
        string text = CTCDecode(host_output.data(), expected_output_elements);
        rec_texts.push_back(text);
        INFO_LOG("Image %d recognition result: %s", idx, text.c_str());
    }

    return SUCCESS;
}

void AscendTextRecognizer::ReleaseResource() {
    // 释放识别模型资源
    if (rec_input_dataset_) {
        aclmdlDestroyDataset(rec_input_dataset_);
        rec_input_dataset_ = nullptr;
    }
    if (rec_output_dataset_) {
        aclmdlDestroyDataset(rec_output_dataset_);
        rec_output_dataset_ = nullptr;
    }
    if (rec_model_desc_) {
        aclmdlDestroyDesc(rec_model_desc_);
        rec_model_desc_ = nullptr;
    }
    if (rec_model_id_) {
        aclmdlUnload(rec_model_id_);
        rec_model_id_ = 0;
    }
    if (rec_input_buffer_) {
        aclrtFree(rec_input_buffer_);
        rec_input_buffer_ = nullptr;
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

//int main() {
//    // 初始化识别器
//    AscendTextRecognizer recognizer(
//            "/home/HwHiAiUser/ocr/models/ch_PP-OCRv4_rec_infer.om",
//            "/home/HwHiAiUser/ocr/models/ppocr_keys_v1.txt"
//    );
//
//    if (recognizer.Init() != SUCCESS) {
//        ERROR_LOG("Failed to initialize recognizer");
//        return 1;
//    }
//
//    // 加载裁剪好的文字区域图像
//    vector<Mat> cropped_images;
//    cropped_images.push_back(imread("/home/HwHiAiUser/ocr/pic/cropped_img.png"));
//
//    // 执行文字识别
//    vector<string> rec_texts;
//    if (recognizer.RecognizeText(cropped_images, rec_texts) == SUCCESS) {
//        for (int i = 0; i < rec_texts.size(); i++) {
//            INFO_LOG("Recognized text %d: %s", i, rec_texts[i].c_str());
//        }
//    }
//
//    return 0;
//}