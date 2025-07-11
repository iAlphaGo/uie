import os
import cv2
import numpy as np

def calculate_rms_contrast(image):
    """
    计算单张图像的RMS对比度
    :param image: 输入图像（支持灰度或彩色）
    :return: RMS对比度值
    """
    if len(image.shape) == 3:  # 彩色图像转为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 转换为浮点数计算
    gray_float = gray.astype(np.float32) / 255.0
    
    # 计算均值和标准差
    mean = np.mean(gray_float)
    std = np.std(gray_float)
    
    # 避免除以零（全黑图像）
    if mean < 1e-6:
        return 0.0
    return std / mean

def main(folder_path):
    """
    计算文件夹内所有图片的RMS对比度平均值
    :param folder_path: 图片文件夹路径
    """
    # 支持的图像格式扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    # 遍历文件夹
    contrast_values = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过非图像文件
        if not filename.lower().endswith(valid_exts):
            continue
        
        try:
            # 读取图像（保留原始通道信息）
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"警告：无法读取文件 {filename}，已跳过")
                continue
            
            # 计算RMS对比度
            contrast = calculate_rms_contrast(img)
            contrast_values.append(contrast)
            
            print(f"处理完成：{filename} \t RMS对比度：{contrast:.4f}")
            
        except Exception as e:
            print(f"处理 {filename} 时发生错误：{str(e)}")
            continue
    
    # 统计结果
    if len(contrast_values) == 0:
        print("未找到可处理的图像文件")
        return
    
    avg_contrast = np.mean(contrast_values)
    print("\n===== 最终结果 =====")
    print(f"处理图片数量：{len(contrast_values)}")
    print(f"平均RMS对比度：{avg_contrast:.4f}")

if __name__ == "__main__":
    # 输入你的图片文件夹路径
    image_folder = r"generated_imagesvmd4msr"
    #C:/Users/lwt/Pictures/generated_images_model62_2
    # 检查路径是否存在
    if not os.path.isdir(image_folder):
        print(f"错误：路径 {image_folder} 不存在或不是文件夹")
    else:
        main(image_folder)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_contrast(image_path):
#     # 读取图像
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # 计算图像的标准差
#     contrast = np.std(image)
#     return contrast

# # 示例使用
# image_path = 'example_image.jpg'  # 替换为你的图像路径
# contrast_value = calculate_contrast(image_path)
# print(f"图像对比度为: {contrast_value}")


#可视化
# def show_images(original_image_path):
#     # 读取原图像
#     original_image = cv2.imread(original_image_path)

#     # 转换为灰度图
#     gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

#     # 显示图像
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     plt.title('原图像')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(gray_image, cmap='gray')
#     plt.title('灰度图像')
#     plt.axis('off')

#     plt.show()

# # 示例使用
# show_images(image_path)