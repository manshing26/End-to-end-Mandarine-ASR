from numpy import average
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, AveragePooling2D, Input

def Simple_cnn(path,freeze=True,load_trained=True):
    input_data = Input(name='the_input_scnn', shape=(1600, 39, 1))

    c1 = Conv1D(32,3,padding="valid",kernel_initializer='glorot_normal')(input_data)
    c2 = Conv1D(32,3,padding="valid",kernel_initializer='glorot_normal')(c1)
    avg_pool1 = AveragePooling2D(pool_size=(2,1))(c2)

    c3 = Conv1D(64,5,padding="valid",kernel_initializer='glorot_normal')(avg_pool1)
    c4 = Conv1D(64,5,padding="valid",kernel_initializer='glorot_normal')(c3)
    avg_pool2 = AveragePooling2D(pool_size=(2,1))(c4)

    c5 = Conv1D(128,7,padding="valid",kernel_initializer='glorot_normal')(avg_pool2)
    c6 = Conv1D(128,7,padding="valid",kernel_initializer='glorot_normal')(c5)
    avg_pool3 = AveragePooling2D(pool_size=(2,1))(c6)

    scnn = Model(input_data,avg_pool3)

    if load_trained:
        scnn.load_weights(path)

    if freeze==True:
        scnn.trainable = False

    return scnn

def Simple_cnn_L(path,freeze=True,load_trained=True):
    input_data = Input(name='the_input_scnn', shape=(1600, 39, 1))

    c1 = Conv1D(64,3,padding="valid",kernel_initializer='glorot_normal')(input_data)
    c2 = Conv1D(64,3,padding="valid",kernel_initializer='glorot_normal')(c1)
    avg_pool1 = AveragePooling2D(pool_size=(2,1))(c2)

    c3 = Conv1D(128,5,padding="valid",kernel_initializer='glorot_normal')(avg_pool1)
    c4 = Conv1D(128,5,padding="valid",kernel_initializer='glorot_normal')(c3)
    avg_pool2 = AveragePooling2D(pool_size=(2,1))(c4)

    c5 = Conv1D(256,7,padding="valid",kernel_initializer='glorot_normal')(avg_pool2)
    c6 = Conv1D(256,7,padding="valid",kernel_initializer='glorot_normal')(c5)
    avg_pool3 = AveragePooling2D(pool_size=(2,1))(c6)

    scnn = Model(input_data,avg_pool3)

    if load_trained:
        scnn.load_weights(path)

    if freeze==True:
        scnn.trainable = False

    return scnn
