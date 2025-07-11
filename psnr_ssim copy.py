import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr_ssim(image_folder1, image_folder2, ext1='.jpg', ext2='.png'):
    image_files1 = os.listdir(image_folder1)
    image_files2 = os.listdir(image_folder2)

    psnr_values = []
    ssim_values = []

    # 提取 folder1 中所有文件的主名（不带后缀）
    base_names1 = {os.path.splitext(f)[0]: f for f in image_files1 if f.endswith(ext1)}
    # 提取 folder2 中所有文件的主名（不带后缀）
    base_names2 = {os.path.splitext(f)[0]: f for f in image_files2 if f.endswith(ext2)}

    # 遍历 folder1 中的文件主名，寻找匹配的 folder2 文件
    for base_name in base_names1:
        if base_name in base_names2:
            # 获取完整文件名（含后缀）
            img_file1 = base_names1[base_name]
            img_file2 = base_names2[base_name]

            img_path1 = os.path.join(image_folder1, img_file1)
            img_path2 = os.path.join(image_folder2, img_file2)

            # 加载图像并检查有效性
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            if img1 is None or img2 is None:
                print(f"警告：无法加载图像对 {img_file1} 和 {img_file2}，跳过")
                continue

            # 统一尺寸
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 归一化到 [0,1]
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0

            # 计算 PSNR（显式指定数据范围）
            try:
                psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)
                psnr_values.append(psnr)
            except:
                print(f"计算 {img_file1} 的 PSNR 失败")
                continue

            # 转换为灰度图（使用 BGR2GRAY）
            try:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            except:
                print(f"转换 {img_file1} 为灰度图失败")
                continue

            # 计算 SSIM（单通道模式）
            try:
                ssim_value = structural_similarity(
                    img1_gray, 
                    img2_gray,
                    data_range=1.0,
                    channel_axis=None
                )
                ssim_values.append(ssim_value)
            except:
                print(f"计算 {img_file1} 的 SSIM 失败")
                continue

    if not psnr_values or not ssim_values:
        raise ValueError("无有效图像对可供计算！请检查文件后缀和主名匹配。")

    return psnr_values, ssim_values

# 示例用法（假设 folder1 为 .jpg，folder2 为 .png）
folder1 = 'images_ssr2_ruie'
folder2 = 'C:/underwater_data/raw-890'

try:
    psnr_values, ssim_values = calculate_psnr_ssim(folder1, folder2, ext1='.jpg', ext2='.png')
    
    for i, (psnr, ssim_val) in enumerate(zip(psnr_values, ssim_values)):
        print(f"Image Pair {i+1}: PSNR = {psnr:.2f}, SSIM = {ssim_val:.4f}")
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Average PSNR = {avg_psnr:.2f}")
    print(f"Average SSIM = {avg_ssim:.4f}")

except Exception as e:
    print(f"错误: {str(e)}")