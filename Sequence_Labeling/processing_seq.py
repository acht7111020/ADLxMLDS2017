import os
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from keras.utils import to_categorical

train_rawData = open("data/mfcc/train.ark","r").read().splitlines()
test_rawData = open("data/mfcc/test.ark","r").read().splitlines()
label_rawData = open("data/label/train.lab","r").read().splitlines()
print('train raw data x', len(train_rawData ))
print('test raw data x', len(test_rawData ))
print('train raw data y', len(label_rawData))

train_dict = {}
for i in range(0,len(train_rawData)):
    tempStr = train_rawData[i]
    tempX = tempStr.split( )
    
    temp = tempX[0].split("_")
    instance = temp[0] + "_" + temp[1]
    frame = int(temp[2])
    features = list(map(float, tempX[1:]))
    
    if instance not in train_dict:
        train_dict[instance] = {}
    
    train_dict[instance][frame-1] = (np.asarray(features).reshape(1,-1)) # normalize

test_dict = {}
for i in range(0,len(test_rawData)):
    tempStr = test_rawData[i]
    tempX = tempStr.split( )
    
    temp = tempX[0].split("_")
    instance = temp[0] + "_" + temp[1]
    frame = int(temp[2])
    features = list(map(float, tempX[1:]))
    
    if instance not in test_dict:
        test_dict[instance] = {}
    
    test_dict[instance][frame-1] = (np.asarray(features).reshape(1,-1)) # normalize

# phone_char48 = open("data/48phone_char.map").read().splitlines()
# phoneToNum = {}
# for x in phone_char48:
#     tempX = x.split('\t')
#     phoneToNum[tempX[0]] = int(tempX[1])

map48_39_raw = open("data/phones/48_39.map").read().splitlines()
map48_39 = {}
for x in map48_39_raw:
    tempX = x.split('\t')
    map48_39[tempX[0]] = tempX[1]

mapping39 = {}
count = 0
label_dict = {}
for i in range(0,len(label_rawData)):
    tempStr = label_rawData[i]
    tempX = tempStr.split(',')
    
    temp = tempX[0].split("_")
    instance = temp[0] + "_" + temp[1]
    frame = int(temp[2])
    
    if instance not in label_dict:
        label_dict[instance] = {}
    label = map48_39[tempX[1]]
    if label not in mapping39:
        mapping39[label] = count
        count += 1
        
    lab = mapping39[label]
    label_dict[instance][frame-1] = to_categorical(lab, 39)[0]
    if i == 0:
        print(label_dict[instance][frame-1])
print(len(train_dict), len(test_dict), len(label_dict))
print(mapping39)
train_data = []
label_data = []
test_data = []
for key, value in train_dict.items():
    train_seq = []
    label_seq = []
    for f in sorted(value):
        train_seq.append(train_dict[key][f])
    for f in sorted(label_dict[key]):
        label_seq.append(label_dict[key][f])
    train_data.append(train_seq)
    label_data.append(label_seq)

test_name_list = []
for key, value in test_dict.items():
    test_seq = []
    test_name_list.append(key)
    # print(key)
    for f in sorted(value):
        test_seq.append(test_dict[key][f])

    test_data.append(test_seq)

print(len(train_data), len(test_data), len(label_data), len(test_name_list))

np.save('data/mfcc/seq_mapping39', mapping39)
with open('data/mfcc/seq_x.dat', 'wb') as f:
	pickle.dump(train_data, f)
with open('data/mfcc/seq_test.dat', 'wb') as f:
	pickle.dump(test_data, f)
with open('data/mfcc/seq_y.dat', 'wb') as f:
	pickle.dump(label_data, f)
with open('data/mfcc/seq_y_file.dat', 'wb') as f:
    pickle.dump(test_name_list, f)
