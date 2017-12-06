## -*- coding: utf-8 -*-
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from process import one_hot
import cv2
data_dir = 'data/processed/' #Change to your directory here

def load_data(data_dir):
    train = pd.read_json(data_dir + 'train.json')

    #Fill 'na' angles with mode
#    train.inc_angle = train.inc_angle.replace('na', 0)
#    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

    return train

train = load_data(data_dir)

print train[0:9]

def color_composite(data):

    w,h = 224,224
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        #Add in to resize for resnet50 use 197 x 197
        rgb = cv2.resize(rgb, (w,h)).astype(np.float32)
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

rgb_train = color_composite(train)

print rgb_train

#
# y_train = train['is_iceberg'].values
# y_train=one_hot(y_train,1604)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_valid, y_train, y_valid = train_test_split(rgb_train, y_train, random_state=2, train_size=0.75)
#
# #from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg16 import VGG16
# from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
# from keras.models import Model
# from keras.optimizers import SGD
#
#
# #Create the model
# #model = simple_cnn()
# #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
#
#
#
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(2, activation='softmax')(x)
#
# model = Model(inputs=base_model.input, outputs=predictions)
#
# for i,layer in enumerate(model.layers):
#     print i,layer.name
# for layer in model.layers[:141]:
#     layer.trainable = True
#
# for layer in model.layers[141:]:
#     layer.trainable = True
#
#
# optimizer_sgd = SGD(lr=0.0001,nesterov=True,decay=0.0005,momentum=0.9)
# #model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=optimizer_sgd,loss='categorical_crossentropy', metrics=['accuracy'])
#
# from keras.preprocessing.image import ImageDataGenerator
# #batch_size = 32
# batch_size = 32
# #Lets define the image transormations that we want
# gen = ImageDataGenerator(horizontal_flip=False,
#                          vertical_flip=False,
#                          width_shift_range=0.1,
#                          height_shift_range=0.1,
#                          zoom_range=0,
#                          rotation_range=0)
#
# # Here is the function that merges our two generators
# # We use the exact same generator with the same random seed for both the y and angle arrays
# def gen_flow_for_one_input(X1, y):
#     genX1 = gen.flow(X1, y, batch_size=batch_size, seed=420)
#     while True:
#         X1i = genX1.next()
#         yield X1i[0], X1i[1]
#
# #Finally create out generator
# gen_flow = gen_flow_for_one_input(X_train, y_train)
#
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# epochs_to_wait_for_improve = 100
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
# checkpoint_callback = ModelCheckpoint('ResNet50_5.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#
# #fit the model
# history=model.fit_generator(gen_flow, validation_data=(X_valid, y_valid), steps_per_epoch=int(np.ceil(len(X_train)/batch_size)), epochs=500, verbose=1,callbacks=[early_stopping_callback, checkpoint_callback])
#
# #画出来误差和准确率
# plt.plot()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
