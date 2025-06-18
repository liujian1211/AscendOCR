import numpy as np
import os
from PIL import Image

# ImageNet标准归一化参数（用于转换为int8前的计算）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

# 量化参数（根据您的模型设置调整）
QUANT_SCALE = 255.0  # 量化比例因子
QUANT_ZERO_POINT = 0  # 零点偏移（通常为0）

def process(input_path):
    try:
        # 1. 读取并调整尺寸
        input_image = Image.open(input_path)
        
        # 保持宽高比的缩放
        original_width, original_height = input_image.size
        ratio = min(256/original_width, 256/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        input_image = input_image.resize((new_width, new_height), resample=Image.BILINEAR)
        
        # 2. 转换为numpy数组并确保RGB顺序
        img = np.array(input_image)
        if img.ndim == 2:  # 灰度图处理
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # 去除alpha通道
            img = img[:, :, :3]
        
        # 3. 中心裁剪
        height, width = img.shape[:2]
        crop_size = min(height, width, 224)
        h_off = max(0, (height - crop_size) // 2)
        w_off = max(0, (width - crop_size) // 2)
        crop_img = img[h_off:h_off+crop_size, w_off:w_off+crop_size, :]
        
        # 4. 二次缩放确保224x224
        if crop_size != 224:
            crop_img = Image.fromarray(crop_img)
            crop_img = crop_img.resize((224, 224), resample=Image.BILINEAR)
            crop_img = np.array(crop_img)
        
        # 5. 颜色通道转换：RGB to BGR
        img_bgr = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        
        # 6. 标准化处理（ImageNet标准）
        img_normalized = (img_bgr - IMAGENET_MEAN) / IMAGENET_STD
        
        # 7. 量化为int8（关键步骤）
        # 计算量化后的值：q = round(f * scale + zero_point)
        img_quantized = np.round(img_normalized * QUANT_SCALE + QUANT_ZERO_POINT)
        img_quantized = np.clip(img_quantized, -128, 127)  # 确保在int8范围内
        img_int8 = img_quantized.astype(np.int8)
        
        # 8. 转换为NCHW格式
        result = img_int8.transpose(2, 0, 1)  # HWC to CHW
        result = np.expand_dims(result, axis=0)  # 添加batch维度
        
        # 9. 保存为二进制文件（int8格式）
        output_name = input_path.rsplit('.', 1)[0] + ".bin"
        result.tofile(output_name)
        
        # 验证数据范围和类型
        print(f"输出文件: {output_name}")
        print(f"数据形状: {result.shape}")  # 应该输出 (1, 3, 224, 224)
        print(f"数据类型: {result.dtype}")  # 应该输出 int8
        print(f"数据范围: [{np.min(result)}, {np.max(result)}]")
        
        return 0
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return 1

if __name__ == "__main__":
    count_ok = 0
    count_ng = 0
    images = os.listdir(r'./')
    dir = os.path.realpath("./")
    
    # 创建输出目录
    output_dir = os.path.join(dir, "quant_data")
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in images:
        if not (image_name.lower().endswith((".bmp", ".dib", ".jpeg", ".jpg", ".jpe",
        ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"))):
            continue
        
        print("start to process image {}....".format(image_name))
        image_path = os.path.join(dir, image_name)
        ret = process(image_path)
        
        if ret == 0:
            print("process image {} successfully".format(image_name))
            count_ok = count_ok + 1
        elif ret == 1:
            print("failed to process image {}".format(image_name))
            count_ng = count_ng + 1
    
    print("{} images in total, {} images process successfully, {} images process failed"
          .format(count_ok + count_ng, count_ok, count_ng))