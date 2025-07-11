"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image,ImageOps
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from uqim_utils import getUIQM


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
#inp_dir = "/home/xahid/datasets/released/EUVP/test_samples/Inp/"
# inp_dir = "D:/FUnIE-GAN-master/output_val"

## UIQMs of the distorted input images 
# inp_uqims = measure_UIQMs(inp_dir)
# print ("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

## UIQMs of the enhanceded output images
#gen_dir = "eval_data/euvp_test/funie-gan/" 
gen_dir = "images_msr2_euvp_ui"#测试图片地址
gen_uqims = measure_UIQMs(gen_dir)
print ("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))


# Input UIQMs >> Mean: 2.5001967896251314 std: 0.6068128623806118
# Enhanced UIQMs >> Mean: 1.9544984733191213 std: 0.4504513395197681  funiegan
# Enhanced UIQMs >> Mean: 3.018949124351306 std: 0.4829042432113816   funiegan_up
# Enhanced UIQMs >> Mean: 3.154220469035835 std: 0.30626654010723914   msr
# Enhanced UIQMs >> Mean: 3.0818917831640023 std: 0.24600738431460453  msrcr
# Enhanced UIQMs >> Mean: 3.013325778884584 std: 0.49610186618323243   GAN73
# Enhanced UIQMs >> Mean: 3.0753195233413826 std: 0.4255104405761761    gan72

#gan_msr  Enhanced UIQMs >> Mean: 3.06515055673897 std: 0.4544940221928065
#msrcr   Enhanced UIQMs >> Mean: 1.7811302573885546 std: 0.34321635241168386
#gan_ssr  Mean: 1.7811302573885546 std: 0.34321635241168386