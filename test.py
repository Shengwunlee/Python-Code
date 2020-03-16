import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))
dataset = pd.read_csv('0604_ALL.csv')
train_Data = pd.read_csv('0604.csv')
# Any results you write to the current directory are saved as output.

x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

print(x)
print(y)
