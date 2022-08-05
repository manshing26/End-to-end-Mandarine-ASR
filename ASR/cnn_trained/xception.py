from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def Xception_trained(path,freeze=True,load_trained=True):
    input_data = Input(name='the_input', shape=(1600, 60, 1))

    xception_model = Xception(
        include_top=False, weights=None, 
        input_tensor = input_data, pooling=None
    )

    xception = Model(input_data,xception_model.output)

    if load_trained:
        xception.load_weights(path)

    if freeze==True:
        xception.trainable = False

    return xception