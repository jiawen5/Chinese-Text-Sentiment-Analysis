
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
np.random.seed(123)
import jieba

import os, time, math, keras
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint

# parameters
batch_size=192 #256
epochs=5
model_type='LSTM'
maxlen = 100 #80

# load data
pos=pd.read_excel('pos.xls',header=None,index=None)
neu=pd.read_excel('neu.xls',header=None,index=None)
neg=pd.read_excel('neg.xls',header=None,index=None)
#label data
pos['mark']= 1
neu['mark']= 0
neg['mark']=-1
# define word seperation function
cw = lambda x: list(jieba.cut(x)) 
pos['words'] = pos[0].apply(cw)
neu['words'] = neu[0].apply(cw)
neg['words'] = neg[0].apply(cw)

d2v_train = pd.concat([pos['words'], neu['words'], neg['words']], ignore_index = True) 

w = [] 
for i in d2v_train:
    w.extend(i)
#count how many times each word occurs
dict = pd.DataFrame(pd.Series(w).value_counts()) 
dict['id']=list(range(1,len(dict)+1))

pn=pd.concat([pos,neu,neg],ignore_index=True)
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)

print("Padding sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

# shufflle data
alen=len(pn['sent'])
index=np.arange(alen)
np.random.shuffle(index)

xshuf=np.array(list(pn['sent']))
xshuf=xshuf[index]
yshuf=np.array(list(pn['mark']))
yshuf=yshuf[index]
tlen=math.floor(0.8*alen)
# all set
xa = xshuf 
ya = yshuf
# training set
x = xa[:tlen:1]
y = ya[:tlen:1]
# test set
xt = xa[tlen::1]
yt = ya[tlen::1]

num_classes=3
y = keras.utils.to_categorical(y, num_classes)
yt = keras.utils.to_categorical(yt, num_classes)
ya = keras.utils.to_categorical(ya, num_classes)


max_features=len(dict)+1
print('Building model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.7))   # LSTM or GRU
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
# model.summary()

# checkpoint save best model
save_dir = os.path.join(os.getcwd(), model_type)
model_name = '%s_epoch.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = []

print('Training model...')
t1=time.clock()
model.fit(x, y, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          validation_data=(xt, yt),
          shuffle=True,
          callbacks=callbacks) 
t2=time.clock()

score = model.evaluate(xt, yt, verbose=0)
print('Total training:%12.6fs,%4ds/epoch; Test loss:%.6f; Test accuracy:%.6f ; %s'%(t2-t1, (t2-t1)/epochs, score[0], score[1], time.asctime(time.localtime(time.time()))) )

logpath=model_type+'_log.txt'
with open(logpath, 'a') as f:
    f.write('\n%s batch size:%4d; Total training:%12.6fs,%4ds/epoch; Test loss:%.6f; Test accuracy:%.6f ; %s'%(model_type, batch_size, t2-t1, (t2-t1)/epochs, score[0], score[1], time.asctime(time.localtime(time.time()))) )
