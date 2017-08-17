# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation

# モデルのつくりはコンストラクタで指定してもいいし add メソッドで追加してもいい
model = Sequential([
    # Dense(32, input_dim=784) でもいい
    Dense(32, input_shape=(784,)),
    Activation('relu'),
])
model.add(Dense(10))
model.add(Activation('softmax'))

# rmsprop: 勾配法の一種 http://qiita.com/tokkuman/items/1944c00415d129ca0ee9#rmsprop
# MSE: Mean Squared Error; 平均二乗誤差
model.compile(optimizer='rmsprop',
              loss='mse')

print('finish')
