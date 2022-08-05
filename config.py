from ASR.file_function.file_func import path_checker, slash, select_bg, output_auto
from ASR.file_function.logic_func import GetLogfbank, GetMfcc_slim, GetMfcc
import ASR.PINYIN_model_architecture
import os

path_checker = path_checker
slash = slash

#################################### model parameter ####################################
GPU = True
VERSION = 'verR1.1c'
OUTPUT_MODE = "C"
DOUBLE_INPUT = False
FEATURE_EXTRACTION = GetLogfbank
FEATURE_LENGTH = FEATURE_EXTRACTION.FEATURE_LENGTH
FEATURE_EXTRACTION_2= GetMfcc
FEATURE_LENGTH_2 = FEATURE_EXTRACTION_2.FEATURE_LENGTH
FEATURE_BG, FEATURE_BG_2 = select_bg(FEATURE_LENGTH,FEATURE_LENGTH_2)
AUDIO_LENGTH = 1600
OUTPUT_LENGTH = 200
LABEL_MAX_STRING_LENGTH = 64
OUTPUT_SIZE, DICT = output_auto(OUTPUT_MODE) ## Size: 1424(pinyin), 4788(word)
VAD_PADDING = 5

## For training
architecture = ASR.PINYIN_model_architecture.resnet_gru2c_lib
AUDIO_SPEED = 1
SPEED_CHANGE = "constant" ## default 'constant'. If set to 'range', audio speed will fluctuate
ADD_NOISE = False
SAVE_STEP = 500
BATCH_SIZE = 5
DATASETS = ['st-cmds','aishell','aidatatang','magicdata','primewords']
EPSILON = 0.05

#################################### path parameter ####################################

abs_path_prefix = os.path.abspath(os.path.dirname(__file__))+slash()
cnn_trained_folder = abs_path_prefix + 'ASR/cnn_trained/'
test_audio = abs_path_prefix + "sample/cn01.wav"
dict_folder = abs_path_prefix + 'ASR/dict/'
bg_folder = abs_path_prefix + 'ASR/bg/'
pinyin_dict = dict_folder + 'dict.txt'
word_dict = dict_folder + 'word2.txt'

data_path = "~/your_wd/audio_folder/"
speech_model_folder = "~/your_wd/model_PINYIN/"
DL_language_model_folder = "~/your_wd/model_Language/"

#################################### TFX parameter ####################################

USE_TFX = False
TFX_ASR_PATH = "http://localhost:8501/v1/models/asr:predict"
TFX_LM_PATH = "http://localhost:8501/v1/models/lm:predict"