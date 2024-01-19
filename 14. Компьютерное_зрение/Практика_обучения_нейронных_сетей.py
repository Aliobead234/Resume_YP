#Обучение многослойной сети [GPU]

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
 
 
def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.
    return features_train, target_train
 
 
def create_model(input_shape):
    model = Sequential()
    
    model.add(Dense(units=500, input_shape=input_shape, activation="relu"))
    model.add(Dense(units=300, activation="relu"))
    model.add(Dense(units=10, activation='softmax'))
    return model
 
 
def train_model(model, train_data, test_data, batch_size=48, epochs=50,
               steps_per_epoch=None, validation_steps=None):
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model

'''Epoch 50/50
60000/60000 - 4s - loss: 0.3248 - acc: 0.8857 - val_loss: 0.4960 - val_acc: 0.8511
10000/10000 - 1s - loss: 0.4960 - acc: 0.8511
'''

#Алгоритм Adam

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense,Conv2D,AvgPool2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    return features_train, target_train


def create_model(input_shape):
    optimizer = Adam()
    model = Sequential()
    
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu',
                 input_shape=(28, 28, 1)))

    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
                     input_shape=(28, 28, 1),strides=1))

    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))

    model.add(Flatten())
    
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dense(10, input_shape=input_shape, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=5,
               steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 

#Результаты обучения

#**Вывод:** обучили модель с точностью 89.6%.

#**Лог:**

'''
<class 'tensorflow.python.keras.engine.sequential.Sequential'>
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 - 8s - loss: 0.5391 - acc: 0.8032 - val_loss: 0.4226 - val_acc: 0.8440
Epoch 2/5
60000/60000 - 6s - loss: 0.3702 - acc: 0.8656 - val_loss: 0.3667 - val_acc: 0.8696
Epoch 3/5
60000/60000 - 6s - loss: 0.3197 - acc: 0.8831 - val_loss: 0.3440 - val_acc: 0.8746
Epoch 4/5
60000/60000 - 6s - loss: 0.2864 - acc: 0.8948 - val_loss: 0.3073 - val_acc: 0.8917
Epoch 5/5
60000/60000 - 6s - loss: 0.2633 - acc: 0.9039 - val_loss: 0.2898 - val_acc: 0.8960
10000/10000 - 1s - loss: 0.2898 - acc: 0.8960
'''

# Свёрточные сети для классификации фруктов
# Постройте и обучите свёрточную нейронную сеть 
# на наборе данных с фруктами.
from tensorflow.keras.layers import Dense,Conv2D,AvgPool2D,MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def load_train(path):
    datagen = ImageDataGenerator(
        rescale=1./255,
		rotation_range=90,
		horizontal_flip=True
    )
    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse',
        seed=12345
    )
    return train_datagen_flow

def create_model(input_shape):
    
    optimizer = Adam(learning_rate=0.0001)
    model = Sequential()
    
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                 input_shape=input_shape))
    
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    
    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))
    
    model.add(Conv2D(filters=20, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))
    
    model.add(Flatten())
    
    model.add(Dense(1406, input_shape=input_shape, activation='relu'))    
    model.add(Dense(351, input_shape=input_shape, activation='relu'))
    model.add(Dense(12, input_shape=input_shape, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
               steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model 


## ResNet
# Обучите свёрточную сеть ResNet в Keras (классификация фруктов по фото)

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np


def load_train(path):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=0.15,
        height_shift_range=0.15
    )
    train_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345
    )
    return train_datagen_flow


def create_model(input_shape):
    optimizer = Adam(learning_rate=0.0001)
    
    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    include_top=False) 
    
    
    model = Sequential()
    
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=1,
               steps_per_epoch=None, validation_steps=None):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model 


## Результаты обучения

'''
**Вывод:** обучили модель с точностью 99.17% в одну эпоху.

**Лог:**

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resnet50 (Model)             (None, 5, 5, 2048)        23587712  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               524544    
_________________________________________________________________
dense_1 (Dense)              (None, 12)                3084      
=================================================================
Total params: 24,115,340
Trainable params: 24,062,220
Non-trainable params: 53,120
_________________________________________________________________


<class 'tensorflow.python.keras.engine.sequential.Sequential'>
Train for 1463 steps, validate for 488 steps
1463/1463 - 272s - loss: 0.1228 - acc: 0.9622 - val_loss: 0.0234 - val_acc: 0.9917
488/488 - 40s - loss: 0.0234 - acc: 0.9917
'''


## обучение модели
'''
Постройте и обучите свёрточную нейронную сеть на датасете с 
фотографиями людей. Добейтесь значения MAE на тестовой выборке 
не больше 8.
'''
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense,Conv2D,AvgPool2D,MaxPooling2D,Flatten


def load_train(path):
    labels = pd.read_csv('/datasets/faces/labels.csv')
    train_datagen = ImageDataGenerator(validation_split=0.25,
                                       rescale=1./255.,
                                       horizontal_flip=True)
    train_datagen_flow = train_datagen.flow_from_dataframe(
            dataframe=labels,
            directory='/datasets/faces/final_files/',
            x_col='file_name',
            y_col='real_age',
            target_size=(224, 224),
            batch_size=32,
            class_mode='raw',
            subset='training',
            shuffle=False,
            seed=12345) 
    
    return train_datagen_flow

def load_test(path):
    labels = pd.read_csv('/datasets/faces/labels.csv')
    validation_datagen = ImageDataGenerator(validation_split=0.25,
                                            rescale=1/255.)    
    val_datagen_flow = validation_datagen.flow_from_dataframe(
            dataframe=labels,
            directory='/datasets/faces/final_files/',
            x_col='file_name',
            y_col='real_age',
            target_size=(224, 224),
            batch_size=32,
            class_mode='raw',
            subset='validation',
            shuffle=False,
            seed=12345) 
    
    return val_datagen_flow


def create_model(input_shape):
    
    #backbone = ResNet50(input_shape=input_shape,
    #                weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
    #                include_top=False)
    optimizer = Adam(learning_rate=0.0001)
    # замораживаем ResNet50 без верхушки
    # Не замораживаем. Данных достаточно
    #backbone.trainable = False

    model = Sequential()
    #model.add(backbone)
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                 input_shape=input_shape))
    
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    
    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))
    
    model.add(Conv2D(filters=20, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(AvgPool2D(pool_size=2, padding='same',strides=2))
    
    model.add(Flatten())
    
    model.add(Dense(1406, input_shape=input_shape, activation='relu'))    
    model.add(Dense(351, input_shape=input_shape, activation='relu'))
    model.add(Dense(12, input_shape=input_shape, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    return model

# age_mae печатает средний mae для предсказаний для каждого возраста
def age_mae(model, val_datagen_flow, validation_steps):

    #Извлекаем реальные значения лет для картинки
    val_datagen_flow.reset()
    real_ages = val_datagen_flow.labels
    #Получаем предсказание модели сколько лет человеку на картинке
    val_datagen_flow.reset()
    predictions = model.predict(val_datagen_flow, steps=validation_steps) 
    df = pd.DataFrame({'Real_Age': real_ages, 'Prediction': predictions.flatten(),'MAE': abs(real_ages-predictions.flatten())})

    # Группировка по real_ages и вычисление среднего отклонения
    grouped_df = df.groupby('Real_Age').agg({'Prediction': ['mean'], 'MAE': ['mean', 'count']})
    grouped_df.columns = ['Mean_Year_Prediction', 'Mean_MAE', 'Image_Count']
    pd.set_option('display.max_rows', None)
    print(grouped_df)


def train_model(model, train_data, test_data, batch_size=None, epochs=50,
               steps_per_epoch=None, validation_steps=None):

    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              batch_size=batch_size,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    age_mae(model, test_data, validation_steps)
    return model

'''
86                   60.574413  25.425587            1
87                   73.651611  13.348389            1
88                   78.708153   9.291847            1
90                   74.605934  15.394066            6
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
60/60 - 10s - loss: 60.4545 - mae: 5.7223
Test MAE: 5.7223
'''