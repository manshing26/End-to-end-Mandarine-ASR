from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, BatchNormalization, Add, Activation, Reshape, Input

def gru_block_2(x,hidden_unit):
    gru_a = GRU(hidden_unit,return_sequences=True,kernel_initializer='he_normal')(x)
    gru_b = GRU(hidden_unit,return_sequences=True,kernel_initializer='he_normal',go_backwards=True)(x)
    gru_add = Add()([gru_a,gru_b])
    gru_add = BatchNormalization()(gru_add)
    gru_add = Activation('relu')(gru_add)
    return gru_add


def Resnet_50_trained(path,freeze=True,load_trained=True):
    input_data_a = Input(name='the_input_a', shape=(1600, 60, 1))

    resnet_model = ResNet50V2(
        include_top=False, weights=None, 
        input_tensor = input_data_a, pooling=None
    )

    resnet = Model(input_data_a,resnet_model.output)

    if load_trained:
        resnet.load_weights(path)

    if freeze==True:
        resnet.trainable = False

    return resnet
