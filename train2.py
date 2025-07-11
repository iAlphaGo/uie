#信息熵特征图与原图融合
import keras
from keras.models import Sequential, Model
from keras.layers import Input
#from keras.src.engine.input_layer import Input
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout, LeakyReLU, ZeroPadding2D, BatchNormalization, concatenate, Reshape, Permute, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D#, Deconv2D 
# from keras.src.layers.convolutional.conv2d import Conv2D
# from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
# from keras.src.layers.reshaping.up_sampling2d import UpSampling2D
from keras.layers import Lambda
import tensorflow as tf
#from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam



from glob import glob
import numpy as np 

import matplotlib.pyplot as plt
import cv2 as cv
import cv2
from PIL import Image
import os
from scipy.stats import entropy  # 导入 entropy 函数
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_ubyte
from skimage.transform import resize
import time

# 记录开始时间
start_time = time.time()
TEST_SIZE = 1000

 
class FeatureEnhancement(tf.keras.layers.Layer):
    # def __init__(self):
    #     super(FeatureEnhancement, self).__init__()
    #     # 定义卷积层用于调整特征图的通道数
    #     self.conv_layers = [
    #         tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same') for _ in range(4)
    #         for _ in range(4)  # 假设您选择4个特征层###
    #     ]
    #     self.final_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')
    def __init__(self, num_feature_maps=4):
        super(FeatureEnhancement, self).__init__()
        # 根据输入的 num_feature_maps 参数动态创建卷积层
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same') for _ in range(num_feature_maps)
        ]
        self.final_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')
    def call(self, original_image, selected_features):
        """
        原始图像: (batch_size, height, width, channels)
        selected_features: 选定的特征图列表，每个特征图的形状为 (batch_size, height, width, channels)
        """
        if len(original_image.shape) == 3:###
            original_image = tf.expand_dims(original_image, axis=0)###
        enhanced_image = original_image

        # 遍历选定的特征图
        for i, feature_map in enumerate(selected_features):
            if len(feature_map.shape) == 3:###
                feature_map = tf.expand_dims(feature_map, axis=0)###
            
            # 调整特征图的通道数为 3
            # feature_map = self.conv_layers[i](feature_map)
            adjusted_feature = self.conv_layers[i](feature_map)
        # 上采样到原始图像尺寸（如果尺寸不一致）
            if adjusted_feature.shape[1:3] != original_image.shape[1:3]:###
                adjusted_feature = tf.image.resize(adjusted_feature, original_image.shape[1:3])###

            # 上采样特征图到原始图像的尺寸
            # feature_map = tf.image.resize(feature_map, size=tf.shape(original_image)[1:3])

            # 将特征图与原始图像相加
            # enhanced_image += feature_map

            # 残差连接
            enhanced_image += adjusted_feature

        # 最终卷积层
        enhanced_image = self.final_conv(enhanced_image)
        return enhanced_image
    
# class FeatureEnhancement(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         # 通道调整层（动态适配输入通道）
#         self.channel_adjust = tf.keras.layers.Conv2D(64, 1, padding='same')
#         # 最终输出层
#         self.final_conv = tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid')

#     def call(self, original_image, selected_features):
#         # 处理原始图像维度
#         if len(original_image.shape) == 3:
#             original_image = tf.expand_dims(original_image, axis=0)  # (1, H, W, C)
#             # 若实际需要 batch_size=8，扩展批次维度
#             original_image = tf.tile(original_image, [8, 1, 1, 1])  # (8, H, W, C)
        
#         enhanced_image = original_image

#         # 处理每个特征图
#         for feature_map in selected_features:
#             # 处理特征图维度
#             if len(feature_map.shape) == 3:
#                 feature_map = tf.expand_dims(feature_map, axis=0)
            
#             # 通道调整
#             adjusted_feature = self.channel_adjust(feature_map)  # 输出通道=64
            
#             # 空间尺寸对齐
#             adjusted_feature = tf.image.resize(
#                 adjusted_feature,
#                 original_image.shape[1:3],  # 严格对齐 H, W
#                 method='bilinear'
#             )
            
#             # 通道数对齐（64 -> 3）
#             adjusted_feature = tf.keras.layers.Conv2D(3, 1)(adjusted_feature)
            
#             # 残差连接
#             enhanced_image += adjusted_feature  # 确保此时形状为 (8, 256, 256, 3)

#         return self.final_conv(enhanced_image)


class GAN(object):
    def __init__(self, train_clear_path, train_noise_path, batch_size,model_path):
        '''
        :param train_clear_path: the path of clear imgs
        :param train_noise_path: the path of noise imgs
        '''
        self.clears_path = train_clear_path
        self.noises_path = train_noise_path
        self.batch_size = batch_size
        self.img_shape = (256, 256, 3)
          # 新增保存模型权重的路径
        self.model_path = model_path

        #optimizers = Adam(0.0002, 0.5)#Adam优化器，用于编译鉴别器（discriminator）和组合模型（combined model）。
        optimizers = Adam(learning_rate=0.0002, beta_1=0.5)
        
        #构建鉴别器
        self.discriminator = self.build_discriminator()
        #编译鉴别器模型，使用二元交叉熵作为损失函数，Adam优化器作为优化器，并且监视模型的准确率。
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer= optimizers ,
                                   metrics=['accuracy'])
               
        #创建特征提取器模型
        self.feature_extractor = self.build_feature_extractor()
                 
        # 
        inputs = Input(shape=self.img_shape)
        feature_output2 = self.feature_extractor(inputs)
        self.combined = Model(inputs,feature_output2 )
        self.combined.compile(loss='binary_crossentropy',
                              optimizer= optimizers ,
                              metrics=['accuracy'])


        #构建生成器
        self.generator = self.build_generator()
        input_shape = Input(shape=self.img_shape)
        conv17_output, feature_output1, encoder_outputs, *ignored= self.generator(input_shape)

        self.discriminator.trainable = False#将鉴别器模型设置为不可训练，因为在训练组合模型时，只需要训练生成器。

        valid = self.discriminator(conv17_output)#鉴别器对生成图像的预测结果，用于计算组合模型的损失。

        self.combined = Model(input_shape,[conv17_output, valid])#组合模型，输入是图像，输出是生成的图像和鉴别器对生成图像的预测结果。
        #编译组合模型
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer = optimizers)
        
        #创建特征提取器模型
        self.feature_extractor = self.build_feature_extractor()
                 
  
 

    def build_generator(self):
        '''
        build the u-net
        :return: autoencoder of u-net
        '''
        inputs = Input(shape=self.img_shape)
        # Block 1
        # 使用32个卷积核每层做两次卷积，一次池化
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)
        
        

        # Block 2
        # 使用64个卷积核每层做两次卷积，一次池化
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

        # Block 3
        # 使用128个卷积核每层做两次卷积，一次池化
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

        # Block 4
        # 使用256个卷积核每层做两次卷积，一次池化
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)
        # Block 5
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)

        conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
        conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        pool7 = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool')(conv7)

        conv8 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
        conv8 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        pool8 = MaxPooling2D((2, 2), strides=(2, 2), name='block8_pool')(conv8)

        # Block 5
        # 使用512个卷积核每层做两次卷积
        conv9 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool8)
        conv9 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        # 保存每一层的特征图
        encoder_outputs = [pool1, pool2, pool3, pool4, pool5, pool6, pool7, pool8, conv9]
        # 1-2
        up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv8])#上采样
        conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up10)
        conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
        # 2-4
        up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv7])
        conv11 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up11)
        conv11 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

        # 4-8
        up12 = concatenate([UpSampling2D(size=(2, 2))(conv11), conv6])
        conv12 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up12)
        conv12 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv12)

        # 8-16
        up13 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv5])
        conv13 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up13)
        conv13 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

        # 16-32
        up14 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv4])
        conv14 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up14)
        conv14 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv14)

        # 32-64
        up15 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv3])
        conv15 = Conv2D(125, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up15)
        conv15 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv15)

        # 64-128
        up16 = concatenate([UpSampling2D(size=(2, 2))(conv15), conv2])
        conv16 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up16)
        conv16 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv16)

        
        # 128-256
        up17 = concatenate([UpSampling2D(size=(2, 2))(conv16), conv1])
        conv17 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up17)
        conv17 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv17)

        conv17 = Conv2D(3, (3, 3), activation='tanh', padding='same', kernel_initializer='he_normal')(conv17)
        feature_output1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv17)


        # 构建模型
        # model = Model(inputs=inputs, outputs=[conv17, feature_output1])
        model = Model(inputs=inputs, outputs=[conv17, feature_output1] + encoder_outputs)
        optimizer1 = Adam(0.00025, 0.5)
        # 编译模型
        model.compile(optimizer = optimizer1, loss='mse')
        model.summary()
        # 返回模型
        return model
    def calculate_entropy(self,feature_map):
        # 将特征图展平为一维数组
        flattened = feature_map.flatten()
        # 将值归一化为总和为1
        flattened = flattened / np.sum(flattened)
        # 计算信息熵
        return entropy(flattened)
    

    def build_feature_extractor(self):
        model = Sequential()
        
        # 第一个卷积层
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
         # 第二个卷积层
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))


        model.summary()
       
            # 创建一个新的模型，从输入到全连接层之前的输出
        feature_output2 = Model(inputs=model.input, outputs=model.layers[-2].output)
        
        return feature_output2



    
    #颜色补偿
    def color_compensation(self, img):
        # print("Entering color_compensation function")
        # print("Input image shape:", img.shape)
        img = np.double(img)
        # print(type(img))
        R = img[:, :, 2]
        G = img[:, :, 1]
        B = img[:, :, 0]
        # 三颜色通道均值，对应 I¯r I¯g I¯b
        Irm = np.mean(R, axis=0)
        Irm = np.mean(Irm)/256.0
        Igm = np.mean(G, axis=0)
        Igm = np.mean(Igm)/256.0
        Ibm = np.mean(B, axis=0)
        Ibm = np.mean(Ibm)/256.0
        a = 1
        Irc = R + a * (Igm-Irm)*(1-Irm)*G  # 补偿红色通道
        Irc = np.array(Irc.reshape(G.shape), np.uint8)
        Ibc = B + a * (Igm-Ibm)*(1-Ibm)*G  # 补偿蓝色通道
        Ibc = np.array(Ibc.reshape(G.shape), np.uint8)

        G = np.array(G, np.uint8)
        img = cv2.merge([Irc, G, Ibc])
        #print(type(img))
        #print(" image shape:", img.shape)
        target_size = (256, 256, 3)

        #img = cv2.resize(img, target_size)
        img = resize(img, target_size) 
        #print(" image shape:", img.shape)
        #print(type(img))
        #show(img, "color_compensation")
        #print("Exiting color_compensation function")
        return img
    
                       

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same",kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer='he_normal'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_train(self, batch_size):
        '''
        加载数据集
        :param batch_size: batch_size=15
        :return: train data
        '''
        # get all the elements of clears_path
        self.clears_lists = glob('%s/*' % (self.clears_path))
        self.noises_lists = glob('%s/*' % (self.noises_path))

        print(len(self.clears_lists))
        print(len(self.clears_lists)-TEST_SIZE)
        index = np.random.randint(1,len(self.clears_lists)-TEST_SIZE, size=batch_size)
        #print(index)
        #save the batch_imgs
        clear_imgs = []
        noise_imgs = []

        for idx in index:
            # get the match imgs of clears and noises
            c_path = os.path.join(train_clear_path, '{}.jpg'.format(idx))
            n_path = os.path.join(train_noise_path, '{}.jpg'.format(idx))      

            # append clear imgs and noise imgs使用 self.imread 方法加载清晰图像和噪声图像，并将它们添加到对应的列表中。
            clear_img = self.imread_pillow(c_path)
            noise_img = self.imread_pillow(n_path)
            if clear_img is None:
                print("Failed to read clear image:", c_path)
            if noise_img is None:
                print("Failed to read noise image:", n_path)
            clear_imgs.append(clear_img)
            noise_imgs.append(noise_img)


        # normalize clear imgs and noise imgs to -1 - 1
        return np.array(clear_imgs) / 127.5 - 1, np.array(noise_imgs) / 127.5 - 1

    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   ###
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return img
    
    def imread_pillow(self, path):
        img = Image.open(path)
        img = img.convert("RGB")  # 将图像转换为RGB格式
        return np.array(img)
    def load_show(self, batch_size=15):
        '''
        load the imgs to show batch_size=1
        :param batch_size: the number of show imgs
        :return: noise imgs , clear imgs wanted to show
        '''
        index = np.random.randint(len(self.clears_lists)-TEST_SIZE, len(self.clears_lists), batch_size)
        clear_imgs = []
        noise_imgs = []
        for idx in index:
            # get the match imgs of clears and noises
            c_path = os.path.join(train_clear_path, '{}.jpg'.format(idx))
            n_path = os.path.join(train_noise_path, '{}.jpg'.format(idx))
            clear_img = self.imread_pillow(c_path)
            noise_img = self.imread_pillow(n_path)
            if clear_img is None:
                print("Failed to read clear image:", c_path)
            if noise_img is None:
                print("Failed to read noise image:", n_path)
            clear_imgs.append(clear_img)
            noise_imgs.append(noise_img)
        return np.array(clear_imgs) / 127.5 - 1, np.array(noise_imgs) / 127.5 - 1





    def evaluate_generated_images(self, epoch,clear_imgs, noise_imgs,gen_imgs,path):
        #r, c = 3, 4
        clear_imgs = 0.5 * clear_imgs + 0.5
        noise_imgs = 0.5 * noise_imgs + 0.5
        gen_imgs= 0.5 * gen_imgs + 0.5
        psnr_values = []
        ssim_values = []

        for i in range(15):
            clear_img = clear_imgs[i]
            noise_img = noise_imgs[i]
            gen_img = gen_imgs[i]

            psnr_value = psnr(clear_img, gen_img, data_range=clear_img.max() - clear_img.min())
            ssim_value = ssim(clear_img, gen_img, multichannel=True)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            # 保存生成的图像
            
            plt.imsave("images/generated_%d_%d.png" % (epoch, i), gen_img)
            with open("psnr_ssim_values.txt", "a") as f:
                f.write("Epoch {}: PSNR = {:.4f}, SSIM = {:.4f}\n".format(epoch, psnr_value, ssim_value))
                
        print("PSNR:", np.mean(psnr_values))
        print("SSIM:", np.mean(ssim_values))
    # 辅助函数：从选定的特征图重构图像
    def reconstruct_from_selected_features(self, initial_image, selected_features):

        
        enhanced_image = initial_image  # 初始设定为生成的图像
        
        for selected_feature in selected_features:
        # 这里可以根据实际情况选择不同的融合策略

            upsampled_feature = tf.image.resize(selected_feature, initial_image.shape[1:3], method='nearest')

            enhanced_image += upsampled_feature / len(selected_features)
            print("Enhanced image shape:", enhanced_image.shape)
            print("Upsampled feature shape:", upsampled_feature.shape)
        return enhanced_image
    def train_on_batch(self, epochs):

        his_loss = []   #创建一个空列表，用于存储每个epoch的损失值
        valid = np.ones((self.batch_size, 1))  #真实样本的标签,self.batch_size是每个批次的样本数量，数组，其中每个元素都是 1
        fake = np.zeros((self.batch_size, 1))  #生成样本的标签，数组每个元素都是0

         
        print("self.batch_size:", self.batch_size)  
        # 初始化特征增强模块
        # feature_enhancement = FeatureEnhancement()

        # for loop to train model循环遍历每个epoch。
        for epoch in range(epochs):
            train_clear, train_noise = self.load_train(batch_size=self.batch_size) ###
            #print("Output image shape:", train_noise.shape)
           
            #训练鉴别器                                                                       
            color_output = self.color_compensation(train_noise)
            # print("color_output shape:", color_output.shape)

            # 获取生成器的所有输出（包括特征图） conv17_output, feature_output1, encoder_outputs, *ignored= self.generator(input_shape)
            conv17_output, feature_output1, *encoder_outputs= self.generator.predict(train_noise)
            generated_image = conv17_output  # 生成的图像
            # feature_ouptut1 = outputs[1]   # 特征图1
            # 计算每一层特征图的信息熵

          ####
            # entropy_values = []
            # for i, feature_map in enumerate(encoder_outputs):
            #     entropy_value = self.calculate_entropy(feature_map)
            #     entropy_values.append((i, entropy_value))
            # # 按信息熵排序，选择前4层
            # entropy_values.sort(key=lambda x: x[1], reverse=True)
            # selected_layers = [entropy_values[i][0] for i in range(4)]
            # # 使用选定的层进行解码
            # selected_features = [encoder_outputs[i] for i in selected_layers]
          ####  

            # 计算信息熵并筛选层
            entropy_values = []
            for i, feature_map in enumerate(encoder_outputs):
                entropy_value = self.calculate_entropy(feature_map)
                entropy_values.append((i, entropy_value))

            # 根据阈值筛选（阈值设为10）
            threshold = 11.1

            selected_layers = [i for i, value in entropy_values if value >= threshold]

            # 空选保护机制
            if not selected_layers:
                print("Warning: No layers meet the entropy threshold, selecting highest entropy layer")
                entropy_values.sort(key=lambda x: x[1], reverse=True)
                selected_layers = [entropy_values[0][0]]

            # 打印调试信息
            # print(f"Layer Entropies: {[round(v,2) for (_,v) in entropy_values]}")
            # print(f"Selected Layers: {selected_layers}")

            # 使用选定的层进行解码
            selected_features = [encoder_outputs[i] for i in selected_layers]

            feature_enhancement = FeatureEnhancement(len(selected_layers))
             # 将特征图与原始图像融合
            enhanced_image2 = feature_enhancement(train_noise, selected_features)
            # print(enhanced_image2.shape)#(1, 256, 256, 3)



            #融合
            # enhanced_image =  tf.math.multiply(enhanced_image2, color_output)#1相乘
            #2最大值融合
            # enhanced_image = tf.maximum(enhanced_image2, color_output) #Average PSNR = 21.00 Average SSIM = 0.9247
            #3混合乘法-加法
            enhanced_image = enhanced_image2 * color_output + 0.5*enhanced_image2 + 0.5*color_output#Average PSNR = 21.30 Average SSIM = 0.9146

            # enhanced_image = feature_enhancement(color_output, selected_features)
            # fake_clear_fused = tf.math.multiply(f2, enhanced_image2)
            # enhanced_image = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(fake_clear_fused)
            # print(" enhanced_imageshape:",  enhanced_image.shape)

 
            
            fake_loss = self.discriminator.train_on_batch(enhanced_image, fake)
            real_loss = self.discriminator.train_on_batch(train_clear, valid)
            d_loss = 0.5 * np.add(fake_loss, real_loss)#将伪造图片和真实图片的损失进行平均，作为鉴别器的总损失

          

            # ---------------------
            #  Train Generator训练生成器
            # ---------------------
            #组合模型（Combined Model）对生成器进行训练，目标是使鉴别器无法区分生成的图片和真实图片，并且尽可能接近清晰图片。
            g_loss = self.combined.train_on_batch(train_noise, [train_clear, valid])  ####

            print("Epoch : %d/%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
            epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
            


            if epoch % 20 == 0:
                clears, noises = self.load_show()
                #noise2clear = self.generator.predict(noises)
                # noise2clear, feature_output1 = self.generator.predict(noises)
                noise2clear, feature_output1, *encoder_outputs= self.generator.predict(noises)

                his_loss.append([epoch, g_loss[1]])
                path = 'results/gan62_2/'
                if not os.path.exists(path):
                    os.makedirs(path)
                # model.save(path+'model_%d.h5' % epoch)
                path2 = 'results/g/'
                if not os.path.exists(path):
                    os.makedirs(path)

               
                
                self.sample_images(epoch, clears, noises, noise2clear, path)

        self.generator.save(os.path.join(self.model_path, 'generator_final_model.h5'))

             

        his_loss = np.array(his_loss)
        fig = plt.figure()
        plt.plot(his_loss[:, 0], his_loss[:, 1])
        plt.title('Plot Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
        fig.savefig(path + "figure.png")

    
    # save imgs
    def sample_images(self, epoch, clear, noise, noise_img2clear_img, path):    
        r, c = 3, 4
        # transform imgs to 0 - 1
        clear = 0.5 * clear + 0.5
        noise = 0.5 * noise + 0.5
        noise_img2clear_img = 0.5 * noise_img2clear_img + 0.5
    
        # get figure and axs
        fig, axs = plt.subplots(r, c)
        #for i in range(c):
        for i in range(min(c, len(clear))):
            # the firs row
            axs[0, i].imshow(clear[i])
            axs[0, i].axis('off')
            # the second row
            axs[1, i].imshow(noise[i])
            axs[1, i].axis('off')
            # the third row
            axs[2, i].imshow(noise_img2clear_img[i])
            axs[2, i].axis('off')
        # save the imgs
    
        fig.savefig(path + "/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':

    # the path of clears and noise
    train_clear_path = 'output_folderB' #'output_folder1'
    train_noise_path = 'output_folderA' #'output_folder2'
    batch_size = 8
    #model_weights_path = 'model_weights'  # 保存模型权重的路径
    model_path = 'model62_2'
    # get the model
    autoencoder = GAN(train_clear_path, train_noise_path, batch_size, model_path)
    # start train, epochs = 10000
    autoencoder.train_on_batch(epochs=300)

end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time:.6f}秒")


