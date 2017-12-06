## -*- coding: utf-8 -*-
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from keras.applications import ResNet50
from tqdm import tqdm
import cv2
import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.applications import ResNet50
import keras
from keras.models import load_model
import datetime as dt
import cv2
import xgboost as xgb

def read_data(file):
    df = pd.read_json(file)
    return df
def color_composite(data):
    model = load_model('ResNet50_3.h5')
    dense1_layer_model=Model(inputs=model.input,outputs = model.get_layer('avg_pool').output)
#    dense1_layer_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    w,h = 224,224
    Y = []
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
        rgb = np.expand_dims(rgb, axis=0)
        y = dense1_layer_model.predict(rgb, batch_size=1)
        y=y.squeeze()   #输出是【1,1,2048】，压缩成【2048】
        Y.append(y)

    Y = np.array(Y)
    return Y
train= read_data('input/train.json')
train_x=color_composite(train)
train_y=train['is_iceberg'].values

from sklearn.cross_validation import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, random_state=1, train_size=0.75)


xgb_val = xgb.DMatrix(X_valid,label=y_valid)
xgb_train = xgb.DMatrix(X_train, label=y_train)
params={
'booster':'gbtree',
'objective': 'binary:logistic', #多分类的问题

'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':5, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.9, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.01, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 5000 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model('xgb2.model') # 用于存储训练出的模型