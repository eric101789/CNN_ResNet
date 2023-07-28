import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import CSVLogger

# 讀取CSV檔案
df = pd.read_csv('dataset2.csv')

# 讀取圖片並進行資料處理
X = []
y = []
for _, row in df.iterrows():
    img = load_img(row['path'], color_mode='rgb', target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # 正規化像素值
    X.append(img_array)
    y.append(row['state'])

# 轉換成NumPy陣列
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# 初始化 VGG16 模型
vgg_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

# 在 VGG16 模型之上搭建分類模型
classifier = tf.keras.Sequential([
    vgg_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# 定義學習率
epoch_size = 100
initial_learning_rate = 0.001
decay_steps = epoch_size // 4
decay_rate = 0.96
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 編譯模型
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('result/train/csv/train_epoch100_vgg16_logs.csv', append=False)

batch_size = 32
history = classifier.fit(X_train,
                         y_train,
                         epochs=epoch_size,
                         batch_size=batch_size,
                         steps_per_epoch=7875 // batch_size,
                         validation_data=(X_val, y_val),
                         validation_steps=525 // batch_size,
                         callbacks=[csv_logger])

classifier.save('model/train_model_epoch100_vgg16')
