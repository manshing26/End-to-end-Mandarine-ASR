from typing import List
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Reshape, Add, Concatenate, Lambda, 
    Activation, LeakyReLU, Conv2D, Conv1D, MaxPooling2D, 
    AveragePooling2D, GRU, BatchNormalization
    )

## Components

def custom_ctc_loss(args):
    y_pred, labels, input_length, label_length = args
		
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def gru_block(x,hidden_unit,activation='relu'):
    '''
    bi-directional gru block with relu activation
    '''
    gru_a = GRU(hidden_unit,return_sequences=True,kernel_initializer='he_normal')(x)
    gru_b = GRU(hidden_unit,return_sequences=True,kernel_initializer='he_normal',go_backwards=True)(x)
    gru_add = Add()([gru_a,gru_b])
    gru_add = BatchNormalization()(gru_add)
    gru_add = Activation(activation)(gru_add)
    return gru_add

def dense_block(x,hidden_unit:List,activation:str):
    for h in hidden_unit:
        x = Dense(h, use_bias=True, kernel_initializer='he_normal')(x)
        if activation.lower() == 'leakyrelu':
            x = LeakyReLU()(x)
        else:
            x = Activation('relu')(x)
    return x

def get_trained_cnn(type_:str,freeze:bool=True,load_trained:bool=True):

    import config

    if type_ == 'resnet':
        from ASR.cnn_trained.resnet import Resnet_50_trained
        return Resnet_50_trained(
            path=f'{config.cnn_trained_folder}resnet/resnet',
            freeze=freeze,load_trained=load_trained)

    elif type_ == 'xception':
        from ASR.cnn_trained.xception import Xception_trained
        return Xception_trained(
            path=f'{config.cnn_trained_folder}xception/xception',
            freeze=freeze,load_trained=load_trained)

    elif type_ == 'scnn':
        from ASR.cnn_trained.s_cnn import Simple_cnn
        return Simple_cnn(
            path=f'{config.cnn_trained_folder}scnn/scnn',
            freeze=freeze,load_trained=load_trained)

    elif type_ == 'scnn_L':
        from ASR.cnn_trained.s_cnn import Simple_cnn_L
        return Simple_cnn_L(
            path=f'{config.cnn_trained_folder}scnn_L/scnn_L',
            freeze=freeze,load_trained=load_trained)

## Single input

def resnet_a3_lib(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    resnet_model = get_trained_cnn(type_='resnet',freeze=False,load_trained=False)

    r_in = resnet_model.input
    r_out = resnet_model.output

    layer_reshape = Reshape((200,-1))(r_out)
    layer_d1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_reshape)
    layer_d1 = Dropout(0.3)(layer_d1)
    layer_d2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d1)
    layer_d2 = Dropout(0.3)(layer_d2)
    layer_d3 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(layer_d2)

    y_pred = Activation('softmax', name='Activation0')(layer_d3)
    model_data = Model(inputs = r_in, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[r_in, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def xception_1(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    input_data = Input(name='the_input', shape=(audio_length, audio_feature_length, 1))

    from tensorflow.keras import applications
    xcept_model = applications.xception.Xception(
        include_top=False, weights=None, 
        input_tensor = input_data, pooling=None
    )
    x_out = xcept_model.output

    layer_reshape = Reshape((200,-1))(x_out) #b4:896,b2:704
    layer_d1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_reshape)
    layer_d2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d1)
    layer_d3 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d2)
    layer_d4 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d3)
    layer_d5 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(layer_d4)

    y_pred = Activation('softmax', name='Activation0')(layer_d5)
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def xception_gru1(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    flag = False

    if flag:
        xcept_model = get_trained_cnn('xception')
        input_data = xcept_model.input
        x_out = xcept_model.output

    else:
        input_data = Input(name='the_input', shape=(audio_length, audio_feature_length, 1))

        from tensorflow.keras import applications
        xcept_model = applications.xception.Xception(
            include_top=False, weights=None, 
            input_tensor = input_data, pooling=None
        )
        x_out = xcept_model.output

    layer_reshape = Reshape((200,-1))(x_out)

    x = gru_block(layer_reshape,128)
    x = gru_block(x,128)
    x = gru_block(x,128)

    layer_d1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(x)
    layer_d2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d1)
    layer_d3 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d2)
    layer_d4 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d3)
    layer_d5 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(layer_d4)

    y_pred = Activation('softmax', name='Activation0')(layer_d5)
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def resnet_gru2b_lib(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    resnet_model = get_trained_cnn('resnet',freeze=False,load_trained=False)

    r_in = resnet_model.input
    r_out = resnet_model.output

    layer_reshape = Reshape((200,-1))(r_out)

    x = gru_block(layer_reshape,128)
    x = gru_block(x,128)
    x = gru_block(x,128)

    layer_d1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(x)
    layer_d1 = Dropout(0.2)(layer_d1)
    layer_d2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d1)
    layer_d2 = Dropout(0.2)(layer_d2)
    layer_d3 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(layer_d2)

    y_pred = Activation('softmax', name='Activation0')(layer_d3)
    model_data = Model(inputs = r_in, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[r_in, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def resnet_gru2c_lib(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    flag = False
    if flag: # transfer resnet
        resnet_model = get_trained_cnn('resnet')
        input_data = resnet_model.input
        r_out = resnet_model.output

    else: # from Application
        input_data = Input(name='the_input_a', shape=(audio_length, audio_feature_length, 1))

        from tensorflow.keras import applications
        resnet_model = applications.resnet_v2.ResNet50V2(
            include_top=False, weights=None, 
            input_tensor = input_data, pooling=None
        )
        r_out = resnet_model.output

    layer_reshape = Reshape((200,-1))(r_out)

    x = gru_block(layer_reshape,128)
    x = gru_block(x,128)
    x = gru_block(x,128)

    layer_d1 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(x)
    layer_d2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d1)
    layer_d3 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d2)
    #layer_d3 = Dropout(0.2)(layer_d3)
    layer_d4 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_d3)
    #layer_d4 = Dropout(0.2)(layer_d4)
    layer_d5 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(layer_d4)

    y_pred = Activation('softmax', name='Activation0')(layer_d5)
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def cnn2_gru_MFCC(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    input_data = Input(name='the_input_scnn', shape=(audio_length, audio_feature_length, 1))

    c1 = Conv1D(64,3,padding="valid",kernel_initializer='glorot_normal')(input_data)
    c2 = Conv1D(64,3,padding="valid",kernel_initializer='glorot_normal')(c1)
    avg_pool1 = AveragePooling2D(pool_size=(2,1))(c2)

    c3 = Conv1D(128,5,padding="valid",kernel_initializer='glorot_normal')(avg_pool1)
    c4 = Conv1D(128,5,padding="valid",kernel_initializer='glorot_normal')(c3)
    avg_pool2 = AveragePooling2D(pool_size=(2,1))(c4)

    c5 = Conv1D(256,7,padding="valid",kernel_initializer='glorot_normal')(avg_pool2)
    c6 = Conv1D(256,7,padding="valid",kernel_initializer='glorot_normal')(c5)
    avg_pool3 = AveragePooling2D(pool_size=(2,1))(c6)

    layer_reshape = Reshape((200,-1))(avg_pool3)

    x = gru_block(layer_reshape,512)
    x = gru_block(x,512)
    x = gru_block(x,512)

    Dense_block = dense_block(x,[512,256],'leakyrelu')

    layer_d5 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(Dense_block)

    y_pred = Activation('softmax', name='Activation0')(layer_d5)
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def cnn3_gru_MFCC(audio_length,audio_feature_length,max_output_size,label_max_string_length):
    '''
    for final output length 400
    '''

    input_data = Input(name='the_input_scnn', shape=(audio_length, audio_feature_length, 1))

    c1 = Conv1D(480,5,padding="valid",strides=1,kernel_initializer='glorot_normal')(input_data)
    # c2 = Conv1D(640,5,padding="valid",strides=1,kernel_initializer='glorot_normal')(c1)
    avg_pool1 = AveragePooling2D(pool_size=(2,1))(c1)

    c3 = Conv1D(480,5,padding="valid",strides=2,kernel_initializer='glorot_normal')(avg_pool1)
    # c4 = Conv1D(640,5,padding="valid",strides=2,kernel_initializer='glorot_normal')(c3)
    avg_pool2 = AveragePooling2D(pool_size=(2,1))(c3)

    layer_reshape = Reshape((400,-1))(avg_pool2)

    x = gru_block(layer_reshape,512)
    x = gru_block(x,512)
    x = gru_block(x,512)
    x = gru_block(x,512)
    x = gru_block(x,512)

    Dense_block = dense_block(x,[512,256],'leakyrelu')

    layer_d5 = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(Dense_block)

    y_pred = Activation('softmax', name='Activation0')(layer_d5)
    model_data = Model(inputs = input_data, outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

## Double input

def resnet_a4_double(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    ################ input_a
    resnet_model = get_trained_cnn(type_='resnet',freeze=False,load_trained=False)

    input_data_a = resnet_model.input
    r_out = resnet_model.output

    layer_reshape_a = Reshape((200,-1))(r_out)
    Dense_block_a = dense_block(layer_reshape_a,[1024],'leakyrelu')

    ################ input_b
    input_data_b = Input(name='the_input_b', shape=(audio_length, 39, 1))

    c1 = Conv1D(32,3,padding="valid")(input_data_b)
    c2 = Conv1D(32,3,padding="valid")(c1)
    avg_pool1 = AveragePooling2D(pool_size=(2,1))(c2)

    c3 = Conv1D(64,3,padding="valid")(avg_pool1)
    c4 = Conv1D(64,3,padding="valid")(c3)
    avg_pool2 = AveragePooling2D(pool_size=(2,1))(c4)

    c5 = Conv1D(128,3,padding="valid")(avg_pool2)
    c6 = Conv1D(128,3,padding="valid")(c5)
    avg_pool3 = AveragePooling2D(pool_size=(2,1))(c6)

    layer_reshape_b = Reshape((200,-1))(avg_pool3)
    Dense_block_b = dense_block(layer_reshape_b,[1024,512],'leakyrelu')

    ############### combine
    combine = Concatenate()([Dense_block_a,Dense_block_b])

    Dense_block_c = dense_block(combine,[512],'leakyrelu')

    layer_d = Dense(max_output_size, use_bias=True, kernel_initializer='he_normal')(Dense_block_c)

    y_pred = Activation('softmax', name='Activation0')(layer_d)
    model_data = Model(inputs = [input_data_a,input_data_b], outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data_a,input_data_b, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def resnet_a5_double(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    ################ input_a
    resnet_model = get_trained_cnn(type_='resnet',freeze=False,load_trained=False)

    input_data_a = resnet_model.input
    r_out = resnet_model.output

    layer_reshape_a = Reshape((200,-1))(r_out)
    Dense_block_a = dense_block(layer_reshape_a,[512],'leakyrelu')

    ################ input_b
    scnn = get_trained_cnn(type_='scnn',freeze=False,load_trained=False)
    input_data_b = scnn.input
    s_out = scnn.output

    layer_reshape_b = Reshape((200,-1))(s_out)
    Dense_block_b = dense_block(layer_reshape_b,[512],'leakyrelu')

    ############### combine
    combine = Concatenate()([Dense_block_a,Dense_block_b])

    Dense_block_c = dense_block(combine,[512,256],'leakyrelu')

    layer_d = Dense(max_output_size, use_bias=True, kernel_initializer='glorot_normal')(Dense_block_c)

    y_pred = Activation('softmax', name='Activation0')(layer_d)
    model_data = Model(inputs = [input_data_a,input_data_b], outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data_a,input_data_b, labels, input_length, label_length], outputs=loss_out)

    return model, model_data

def resnet_a6_double(audio_length,audio_feature_length,max_output_size,label_max_string_length):

    ################ input_a
    resnet_model = get_trained_cnn(type_='resnet',freeze=False,load_trained=False)

    input_data_a = resnet_model.input
    r_out = resnet_model.output

    layer_reshape_a = Reshape((200,-1))(r_out)
    Dense_block_a = dense_block(layer_reshape_a,[512],'leakyrelu')

    ################ input_b
    scnn = get_trained_cnn(type_='scnn_L',freeze=False,load_trained=False) # L size 1d cnn
    input_data_b = scnn.input
    s_out = scnn.output

    layer_reshape_b = Reshape((200,-1))(s_out)
    Dense_block_b = dense_block(layer_reshape_b,[512],'leakyrelu')

    ############### combine
    combine = Concatenate()([Dense_block_a,Dense_block_b])

    Dense_block_c = dense_block(combine,[512,256],'leakyrelu')

    layer_d = Dense(max_output_size, use_bias=True, kernel_initializer='glorot_normal')(Dense_block_c)

    y_pred = Activation('softmax', name='Activation0')(layer_d)
    model_data = Model(inputs = [input_data_a,input_data_b], outputs = y_pred)

    labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(custom_ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data_a,input_data_b, labels, input_length, label_length], outputs=loss_out)

    return model, model_data
