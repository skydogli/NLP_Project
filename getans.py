import numpy as np
import os
import sys
# 加载 .npy 文件
#pth="log/cycle/gpt-4o-mini-easy-20241213---23-11-skydogli2/"
#pth="log/connectivity/gpt-4o-mini-hard-20241215---15-40-skydogli/"
pth="log/topology/gpt-4o-mini-easy-20241216---18-17-skydogli/"

data = np.load(pth+'answer.npy', allow_pickle=True)
# 输出到文件 answer.txt
with open('answer.txt', 'w') as f:
    for i in data:
        f.write(i+'\n')


data = np.load(pth+'res.npy', allow_pickle=True)

with open('res.txt', 'w') as f:
    for i in data:
        f.write(str(i)+'\n')