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
for dirname, _, filenames in os.walk('UTKFace'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # imgfiles = [x for x in filename.namelist() if re.search(r"^UTKFace.*jpg$", x)]
# # 画像ファイルパスのみ取得
# imgfiles = [ x for x in z.namelist() if re.search(r"^UTKFace.*jpg$", x)]

X=[]
Y=[]
def parse_filepath(filepath):
    try:
        for fil in filepath:
            # ZIPから画像読み込み
            image = Image.open(fil)
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = fil.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            file = os.path.basename(fil)
            file_split = [i for i in file.split('_')]
            X.append(data)
            Y.append(int(file_split[0]))
    except Exception as e:  # いくつか欠損値があるので例外処理をしておく
        print(filepath)
        return None, None, None

# filename.close()
parse_filepath(filenames)

X = np.array(X)
Y = np.array(Y)

del filename

print(X.shape, Y.shape)
