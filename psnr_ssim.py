import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def calculate_pdnr_ssim(image_folder1, image_folder2):
    image_files1 = os.listdir(image_folder1)
    image_files2 = os.listdir(image_folder2)

    pdnr_values = []
    ssim_values = []

    for img_file1 in image_files1:
        if img_file1 in image_files2:
            img_path1 = os.path.join(image_folder1, img_file1)
            img_path2 = os.path.join(image_folder2, img_file1)

            # Load images
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            # Resize images if they have different sizes (optional step)
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Convert to float32 and to grayscale if necessary
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0

            # Calculate PSNR
            pdnr = peak_signal_noise_ratio(img1, img2)
            pdnr_values.append(pdnr)

             # 将图像转换为灰度图以进行 SSIM 计算 （假设是 RGB 图像）
            original_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            enhanced_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            # Calculate SSIM
            # ssim_value, _ = ssim(img1, img2, full=True)
            # ssim_value = structural_similarity(enhanced_gray, original_gray )
            ssim_value, _ = structural_similarity(enhanced_gray, original_gray, full=True, data_range=1.0)
            ssim_values.append(ssim_value)

    return pdnr_values, ssim_values

# 示例用法
folder1 = 'images_520ssr2'# 第一个文件夹路径
folder2 = 'C:/underwater_data/EUVP/Paired/underwater_scenes/validation'  # 第二个文件夹路径

pdnr_values, ssim_values = calculate_pdnr_ssim(folder1, folder2)

# 打印结果 
for i, (pdnr, ssim_value) in enumerate(zip(pdnr_values, ssim_values)):
    print(f"Image {i + 1}: PSNR = {pdnr:.2f}, SSIM = {ssim_value:.4f}")


# 计算平均值
avg_pdnr = np.mean(pdnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR = {avg_pdnr:.2f}")
print(f"Average SSIM = {avg_ssim:.4f}")
