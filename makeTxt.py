import os
import random
trainval_percent = 0.8  #训练集+验证集占总比例多少
train_percent = 0.7 #训练集占验证集+训练集比例多少
xmlfilepath = 'data/custom/Annotations'
txtsavepath = 'data/custom/ImageSets'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('data/custom/ImageSets/trainval.txt', 'w')
ftest = open('data/custom/ImageSets/test.txt', 'w')
ftrain = open('data/custom/ImageSets/train.txt', 'w')
fval = open('data/custom/ImageSets/valid.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
