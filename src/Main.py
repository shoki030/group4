import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %config InlineBackend.figure_formats = {'png', 'retina'}
import os, zipfile, io, re
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
# 画像入力サイズ
image_size=100
import os
# ZIP読み込み
z = zipfile.ZipFile('testdata.zip')
# 画像ファイルパスのみ取得
imgfiles = [ x for x in z.namelist() if re.search(r"^testdata.*jpg$", x)]

X=[]
Y=[]

for imgfile in imgfiles:
    # ZIPから画像読み込み
    image = Image.open(io.BytesIO(z.read(imgfile)))
    # RGB変換
    image = image.convert('RGB')
    # リサイズ
    image = image.resize((image_size, image_size))
    # 画像から配列に変換
    data = np.asarray(image)
    file = os.path.basename(imgfile)
    file_split = [i for i in file.split('_')]
    X.append(data)
    Y.append(int(file_split[0]))

z.close()

X = np.array(X)
Y = np.array(Y)

del z, imgfiles

print(X.shape, Y.shape)

