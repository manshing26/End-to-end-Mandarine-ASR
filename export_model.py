'''
export part of the trained model as the base model for further usage
'''
from ASR.PINYIN_model import ModelSpeech
from ASR.PINYIN_model_architecture import get_trained_cnn

import config

if __name__=="__main__":

    datapath = config.data_path
    type_ = 'resnet'
    test = True

    ms = ModelSpeech()
    ms.LoadModel_Weight()

    ms.export_CNN_component(
        folder=config.cnn_trained_folder, 
        type_= type_,
        bottom='the_input_a', 
        top='post_relu'
        )

    print('[info] Successfully output the CNN_component')

    if test:

        rn = get_trained_cnn(type_ = type_)
        rn.summary()