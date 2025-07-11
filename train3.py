#主要程序
import keras
from keras.models import Sequential, Model
from keras.layers import Input
#from keras.src.engine.input_layer import Input
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout, LeakyReLU, ZeroPadding2D, BatchNormalization, concatenate, Reshape, Permute, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D#, Deconv2D 
from keras.layers import Lambda
import tensorflow as tf
#from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

from wavelet import WaveletTransform, load_image, tensor_to_image
import torch
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
from tools import measure_time,eps,gauss_blur,simplest_color_balance
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
    
                       

    def retinex_SSR(self,img,sigma=250):#sigma=15.80,250
    
        if len(img.shape)==2:
            img=img[...,None]
        ret=np.zeros(img.shape,dtype='uint8')
        for i in range(img.shape[-1]):
            channel=img[...,i].astype('double')
            S_log=np.log(channel+1)
            gaussian=gauss_blur(channel,sigma)
            #gaussian=cv2.filter2D(channel,-1,get_gauss_kernel(sigma)) #conv may be slow if size too big
            #gaussian=cv2.GaussianBlur(channel,(0,0),sigma) #always slower
            L_log=np.log(gaussian+1)
            r=S_log-L_log
            R=r #R=np.exp(r)?
            mmin=np.min(R)
            mmax=np.max(R)
            stretch=(R-mmin)/(mmax-mmin)*255 #linear stretch
            ret[...,i]=stretch
        return ret.squeeze()
    def retinex_MSR(self,img,sigmas=[15,80,250],weights=None):
        if weights==None:
            weights=np.ones(len(sigmas))/len(sigmas)
        elif not abs(sum(weights)-1)<0.00001:
            raise ValueError('sum of weights must be 1!')
        ret=np.zeros(img.shape,dtype='uint8')
        if len(img.shape)==2:
            img=img[...,None]
        for i in range(img.shape[-1]):
            channel=img[...,i].astype('double')
            r=np.zeros_like(channel)
            for k,sigma in enumerate(sigmas):
                r+=(np.log(channel+1)-np.log(gauss_blur(channel,sigma,)+1))*weights[k]
            mmin=np.min(r)
            mmax=np.max(r)
            stretch=(r-mmin)/(mmax-mmin)*255
            ret[...,i]=stretch
        return ret   
    def retinex_MSRCR(self, img, sigmas=[12, 80, 250], s1=0.01, s2=0.01):
        alpha = 125
        ret = np.zeros(img.shape, dtype='uint8')
        csum_log = np.log(np.sum(img, axis=3).astype('double') + 1)
        # msr = self.retinex_MSR(img, sigmas)
        msr = self.MultiScaleRetinex(img, sigmas)
        # print("msr.shape:",msr.shape)
        for i in range(img.shape[0]):  # 遍历每一张图像
             for j in range(img.shape[-1]):
                channel = img[i, ..., j].astype('double')
                # print("channel.shape:",channel.shape)
                r = (np.log(alpha * channel + 1) - csum_log[i]) * msr[i, ..., j]
                stretch = simplest_color_balance(r, s1, s2)
                ret[i, ..., j] = stretch
        return ret

    def MultiScaleRetinex(self,img,sigmas=[15,80,250],weights=None,flag=True):

        if weights==None:
            weights=np.ones(len(sigmas))/len(sigmas)
        elif not abs(sum(weights)-1)<0.00001:
            raise ValueError('sum of weights must be 1!')
        r=np.zeros(img.shape,dtype='double')
        img=img.astype('double')
        for i,sigma in enumerate(sigmas):
            r+=(np.log(img+1)-np.log(gauss_blur(img,sigma)+1))*weights[i]
        if flag:
            mmin=np.min(r,axis=(0,1),keepdims=True)
            mmax=np.max(r,axis=(0,1),keepdims=True)
            r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant
            r=r.astype('uint8')
        return r
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

        # print(len(self.clears_lists))
        # print(len(self.clears_lists)-TEST_SIZE)
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
        his_loss = []
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            train_clear, train_noise = self.load_train(batch_size=self.batch_size)
            
          
        
        # 1. VMD分解与模态分离 ----------------------------------------
            wavelet_transform = WaveletTransform(alpha=5000, tau=0.25, K=5, DC=1, init=1, 
                                           tol=None, eps=2.2204e-16, max_iter=500, 
                                           scale=1, dec=True, transpose=True)
            train_noise_tensor = torch.from_numpy(train_noise).float()
            output_tensor = wavelet_transform(train_noise_tensor)
            # print("[Debug]output_tensor:", output_tensor.shape)#output_tensor: torch.Size([5, 8, 256, 256, 3])
            # 步骤1：调整维度顺序为 [batch_size, K, H, W, C]
            output_tensor = output_tensor.permute(1, 0, 2, 3, 4)  # [8,5,256,256,3]
            
            #  步骤2：计算模态-通道能量
            energies = torch.mean(output_tensor**2, dim=(2,3))  # [batch, K]
            mean_energy = torch.mean(energies)
            threshold = 0.1 * mean_energy
            # 步骤3：创建三维掩码
            high_mask = (energies >= threshold)  # [8,5,3]
            low_mask = ~high_mask

            # 步骤4：扩展掩码维度以匹配output_tensor
            high_mask = high_mask.unsqueeze(2).unsqueeze(3)  # [8,5,1,1,3]
            low_mask = low_mask.unsqueeze(2).unsqueeze(3)    # [8,5,1,1,3]
          
            # 步骤5：重构高/低能量信号
            high_energy_tensor = (output_tensor * high_mask).sum(dim=1)
            low_energy_tensor = (output_tensor * low_mask).sum(dim=1)
            
            # high_mask = torch.where(energies >= threshold)[0]
            # low_mask = ~high_mask
            # high_energy_tensor = output_tensor[:, high_mask, :, :].sum(dim=1)
            # low_energy_tensor = output_tensor[:, low_mask, :, :].sum(dim=1)

            # 转换到TensorFlow张量
            high_energy_tf = tf.convert_to_tensor(high_energy_tensor.numpy())
            low_energy_tf = tf.convert_to_tensor(low_energy_tensor.numpy())
        
            # 2. 双路径处理 ------------------------------------------------
            # 生成器处理高能量部分
            conv17_output, feature_output1, *encoder_outputs = self.generator.predict(high_energy_tf)
            generated_high = conv17_output
            
            # 计算信息熵并筛选层
            entropy_values = []
            for i, feature_map in enumerate(encoder_outputs):
                entropy_value = self.calculate_entropy(feature_map)
                entropy_values.append((i, entropy_value))
            # 根据阈值筛选（阈值设为10）
            threshold = 10.7
            selected_layers = [i for i, value in entropy_values if value >= threshold]
            # 空选保护机制
            if not selected_layers:
                print("Warning: No layers meet the entropy threshold, selecting highest entropy layer")
                entropy_values.sort(key=lambda x: x[1], reverse=True)
                selected_layers = [entropy_values[0][0]]

            # 使用选定的层进行解码
            selected_features = [encoder_outputs[i] for i in selected_layers]
            feature_enhancement = FeatureEnhancement(len(selected_layers))
             # 将特征图与原始图像融合
            enhanced_high = feature_enhancement(generated_high, selected_features)
            # print(enhanced_image2.shape)#(1, 256, 256, 3)

            # Retinex处理低能量部分
            color_output1 = self.retinex_SSR(low_energy_tf.numpy())  # 需确保输入格式兼容
            processed_low = self.color_compensation(color_output1)
        
            # 3. 特征融合 -------------------------------------------------
            # 调整维度匹配 (假设processed_low形状为 [batch, H, W, C])
            processed_low = tf.image.resize(processed_low, enhanced_high.shape[1:3])
            enhanced_image = tf.add(enhanced_high * 0.7, processed_low * 0.3)  # 1加权融合
            
            # 2自适应权重融合（根据能量比例）
            high_weight = tf.reduce_mean(high_energy_tf) / (tf.reduce_mean(high_energy_tf) + tf.reduce_mean(low_energy_tf))
            final_output = tf.add(enhanced_high * high_weight, 
                             processed_low * (1 - high_weight))
        
            # 4. 训练流程调整 ----------------------------------------------
            # 鉴别器训练
            fake_loss = self.discriminator.train_on_batch(enhanced_image, fake)
            real_loss = self.discriminator.train_on_batch(train_clear, valid)
            d_loss = 0.5 * np.add(fake_loss, real_loss)
        
            # 生成器训练（需修改combined模型的输入）
            # g_loss = self.combined.train_on_batch(
            #     [high_energy_tf, low_energy_tf],  # 同时传递双路径输入
            #     [train_clear, valid]              # 保持原目标
            # )

            g_loss = self.combined.train_on_batch(train_noise, [train_clear, valid])
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.3f}, acc: {100*d_loss[1]:.1f}%] [G loss: {g_loss[0]:.3f}, mse: {g_loss[1]:.3f}]")
            


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
    model_path = 'modelvmd8_ssr'
    # get the model
    autoencoder = GAN(train_clear_path, train_noise_path, batch_size, model_path)
    # start train, epochs = 10000
    autoencoder.train_on_batch(epochs=1000)

end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time:.6f}秒")


