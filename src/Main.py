"""顔画像から年齢を予測し、年齢確認を促進するプログラム
UTKFace.zipを使用(https://susanqq.github.io/UTKFace/)。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

"""データの取得

Argments:
    image_size:　画像の入力サイズ。
    z:　zipファイルの読み込み。
    imgfiles:　画像のファイルパスの取得。
    X:　画像から配列に変換したものをリストとして保持。
    Y:　ファイル名を"_"で分割し一番最初の年齢だけをリストとして保持。
    test_size: テストデータの数を決める。

"""

image_size = 100

z = zipfile.ZipFile('UTKFace.zip')
imgfiles = [x for x in z.namelist() if re.search(r"^UTKFace.*jpg$", x)]

X = []
Y = []

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

# trainデータとtestデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    random_state=0,
    test_size=0.2
)
del X, Y
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# データ型の変換＆正規化
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# trainデータからvalidデータを分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    random_state=0,
    test_size=0.2
)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

"""モデルの構築
KerasのXceptionを使用する。
Argments:
    include_top:　ネットワーク出力層の全結合層を除去(False)。
    early_stopping:　過学習を防ぐ関数。

"""
base_model = Xception(
    include_top=False,
    weights="imagenet",
    input_shape=None
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1)(x)
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False: os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    save_freq=3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)

# log for TensorBoard
logging = TensorBoard(log_dir="log/")
# RMSE
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


"""モデル学習
全135層。
Xceptionの特性より、浅い層は固定し深い層を学習。
Argments:
    model:　ネットワークの定義。

"""

model = Model(inputs=base_model.input, outputs=predictions)

# 108層までfreeze
for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalizationのfreeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

# 109層以降、学習させる
for layer in model.layers[108:]:
    layer.trainable = True

# layer.trainableの設定後にcompile
model.compile(
    optimizer=Adam(),
    loss=root_mean_squared_error,
)

hist = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=X_train.shape[0] // 32,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping, reduce_lr],
    shuffle=True,
    verbose=1
)

"""学習曲線のプロット
Argments:
    figsize:　グラフのサイズ。

"""
plt.figure(figsize=(18, 6))

# loss
plt.subplot(1, 2, 1)
plt.plot(hist.history["loss"], label="loss", marker="o")
plt.plot(hist.history["val_loss"], label="val_loss", marker="o")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("")
plt.legend(loc="best")
plt.grid(color='gray', alpha=0.2)

plt.show()

"""モデル評価
Argments:
    score:　損失関数。

"""
score = model.evaluate(X_test, y_test, verbose=1)
print("evaluate loss: {}".format(score))

"""モデルの保存
Argments:
    model_dir:　ディレクトリのパスの指定。
"""
model_dir = './model/'
if os.path.exists(model_dir) == False: os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

# optimizerのない軽量モデルを保存
model.save(model_dir + 'model-opt.hdf5', include_optimizer=False)

"""モデルの予測
testデータ30件の画像と正解値＆予測値を出力する。
予測値の15歳から40歳までを年齢確認するように"o"で出力する。
Argments:
    preds: testデータ30件の予測値。
    pred:　予測年齢。
    tune:　実際年齢。

"""

preds = model.predict(X_test[0:30])


plt.figure(figsize=(16, 6))
for i in range(8):
    plt.subplot(3, 10, i + 1)
    plt.axis("off")
    pred = round(preds[i][0], 1)
    true = y_test[i]
    if 15 < pred < 40:
        plt.title("O" + '\n' + str(true) + '\n' + str(pred), color="red")
    else:
        plt.title("X" + '\n' + str(true) + '\n' + str(pred))
    plt.imshow(X_test[i])
plt.show()