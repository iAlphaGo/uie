#得到了模型结构和权重generator_final_weights.h5
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# Function to preprocess an image 处理图片函数
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 127.5 - 1  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to generate clear image from noise生成器处理图片函数
def generate_clear_image(generator, noise_img):
    #generated_img, _ = generator.predict(noise_img)
    generated_img, *_ = generator.predict(noise_img)  # 忽略其他输出
    generated_img = 0.5 * generated_img + 0.5  # Denormalize to [0, 1]
    return generated_img[0]

# 模型权重，测试图片地址
model_weights_path = 'model520ssr2'  
# test_noise_image_path = 'validation'  
test_noise_image_path = 'validation'
output_path = 'images_520ssr2'  


if not os.path.exists(output_path):
    os.makedirs(output_path)


if os.path.exists(test_noise_image_path):
    # Load the generator model
    generator = load_model(os.path.join(model_weights_path, 'generator_final_model.h5'))

    # Get a list of all files in the folder
    image_files = os.listdir(test_noise_image_path)
    
    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(test_noise_image_path, image_file)
        test_noise_img = preprocess_image(image_path)
        
        # Generate clear image from the noise image using the generator
        generated_clear_img = generate_clear_image(generator, test_noise_img)

        #获取原来的名字
        filename_without_extension = os.path.splitext(image_file)[0]
        
        # Save the generated image
        generated_image_path = os.path.join(output_path, f'{filename_without_extension}.jpg')
        plt.imsave(generated_image_path, generated_clear_img)

        # Display the generated image (optional)
        # plt.imshow(generated_clear_img)
        # plt.axis('off')
        # plt.title('Generated Clear Image')
        # plt.show()

else:
    print(f"Error: Folder '{test_noise_image_path}' does not exist.")
