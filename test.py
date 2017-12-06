import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

plt.rcParams['figure.figsize'] = 10, 10

train = pd.read_json('./data/processed/train.json')
test = pd.read_json('./data/processed/test.json')


