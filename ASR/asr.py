import config
from ASR.file_function.file_func import *
from ASR.file_function.logic_func import *

import os
import json
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

class asr_base():

    slash = config.slash()
    feature_method = config.FEATURE_EXTRACTION
    feature_length = config.FEATURE_LENGTH
    feature_method_2 = config.FEATURE_EXTRACTION_2
    feature_length_2 = config.FEATURE_LENGTH_2
    audio_length = config.AUDIO_LENGTH
    label_max_string_length = config.LABEL_MAX_STRING_LENGTH
    output_size = config.OUTPUT_SIZE
    vad_padding = config.VAD_PADDING
    list_symbol_dic_1424 = GetSymbolList(config.pinyin_dict)
    list_word = GetWordList(config.word_dict)

    @staticmethod
    def _cal_slice(shape0):
        if (shape0 % 1600) == 0:
            slice = int(shape0/1600)
        else:
            slice = int(shape0//1600)+1

        return slice

    @staticmethod
    def _cut_slice(data_input,num_slice):
        r = []
        
        start = 0
        end = 1600
        for i in range(num_slice):
            if i != (num_slice-1):
                r.append(data_input[start:end,:])
                start += 1600
                end += 1600
            else:
                r.append(data_input[start:,:])
        return r

    @staticmethod
    def timestamp(base_pred,empty_num,padding=5): #ratio
        
        max_prob = np.argmax(base_pred,axis=2)#[0] # non-speech = 4787
        max_prob = max_prob.reshape(max_prob.shape[0]*max_prob.shape[1])

        distinguish = []

        # turn to 0/1
        for s in max_prob:
            if s == empty_num:
                distinguish.append(0)
            else:
                distinguish.append(1)

        # tuning
        distinguish_2 = [0]*len(distinguish)

        for idx,s in enumerate(distinguish):
            if s == 1:

                if idx > padding: # no leakage, pad previous slice
                    distinguish_2[idx-padding:idx+1] = [1]*(padding+1)
                else:
                    distinguish_2[0:(idx+1)] = [1]*(idx+1)
                
                if idx+padding < len(distinguish)-1: 
                    distinguish_2[idx:idx+padding+1] = [1]*(padding+1) # 194+4 < 200-1 --> 194:199(5) = 4+1(5)
                else:
                    distinguish_2[idx:] = [1]*(len(distinguish_2)-idx) # 196+4 > 200-1 --> 196:(4) = 200-196(4)

        timestamp = []
        temp = []
        true_flag = False
        for idx,s in enumerate(distinguish_2):
            if s == 1:
                if true_flag == False:
                    temp.append((idx))
                    true_flag = True

            if s == 0:
                if true_flag == True:
                    temp.append(((idx+1)))
                    true_flag = False
                    timestamp.append(temp)
                    temp = []

        if true_flag == True and len(temp)==1:
            temp.append(idx)
            timestamp.append(temp)

        return timestamp

    def Get_data_input(self, wavsignal, fs, channel=0): #double
        data_input = self.feature_method.feature(wavsignal,fs,channel=channel)
        data_input = np.array(data_input, dtype = np.float)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)

        data_input_2 = None
        if config.DOUBLE_INPUT:
            data_input_2 = self.feature_method_2.feature(wavsignal, fs,channel=channel)
            data_input_2 = np.array(data_input_2, dtype = np.float)
            data_input_2 = data_input_2.reshape(data_input_2.shape[0],data_input_2.shape[1],1)
        
        return data_input,data_input_2

    def get_base_predict(self,input_):

        if config.USE_TFX:
            if config.DOUBLE_INPUT:
                i = {"instances":[{
                    'the_input_a':input_[0][0].tolist(),
                    'the_input_scnn':input_[1][0].tolist()
                    }]}
            else:
                i = {"instances":input_.tolist()}

            r = requests.post(config.TFX_ASR_PATH, data=json.dumps(i))
            base_pred = json.loads(r.text)['predictions']
            base_pred = np.array(base_pred)
            base_pred = base_pred[:,:,:] #shape: (batch,audio_len_compress,categories)
        else:
            base_pred = self.base_model.predict(x=input_)

        return base_pred

    def Predict_single(self,data_input_ls,input_len):
        batch_size = len(data_input_ls)

        x = np.zeros((batch_size,self.audio_length,self.feature_length,1),dtype=np.float)

        for i in range(batch_size):

            x[i]=self.bg
            x[i,:(len(data_input_ls[i]))] = data_input_ls[i] #forward
            # x[i,-(len(data_input_ls[i])):] = data_input_ls[i] #backward
            if self.feature_method.USE_CMVN == True:
                x[i] = CMVN(x[i])
        
        base_pred = self.get_base_predict(input_=x)
        
        ## Cut slice by VAD
        timestamp = self.timestamp(base_pred,empty_num=(self.output_size-1),padding=self.vad_padding)
        base_pred_r=base_pred.reshape(base_pred.shape[0]*base_pred.shape[1],base_pred.shape[2])
        num_slice = len(timestamp)
        max_len = input_len
        if num_slice != 0:
            max_len = max([(t[1]-t[0]) for t in timestamp])
        new = np.zeros((num_slice,max_len,self.output_size))
        new[:,:,-1] = 1
        for idx,t in enumerate(timestamp):
            len_t = t[1]-t[0]
            new[idx,0:len_t] = base_pred_r[t[0]:t[1],:]

        in_len_new = np.zeros((num_slice),dtype=np.int32)
        in_len_new[:] = max_len
        
        ## Decode
        r = K.ctc_decode(new, in_len_new, greedy = True, beam_width=100, top_paths=1)
        r1 = K.get_value(r[0][0]) #(batch,200)
        
        pinyin_sequence = []

        for i in range(num_slice):
            pinyin_sequence.append(
                [[num for num in r1[i] if num != -1],
                list(map(lambda x:x*80,timestamp[i]))]
                ) # exclude -1
        
        return pinyin_sequence #num

    def Predict_double(self,data_input_ls,input_len):
        batch_size = len(data_input_ls[0])

        x = np.zeros((batch_size,self.audio_length,self.feature_length,1),dtype=np.float)
        x2 = np.zeros((batch_size,self.audio_length,self.feature_length_2,1),dtype=np.float)

        for i in range(batch_size):

            x[i]=self.bg
            x[i,:(len(data_input_ls[0][i]))] = data_input_ls[0][i] #forward
            if self.feature_method.USE_CMVN == True:
                x[i] = CMVN(x[i])

            x2[i]=self.bg_2
            x2[i,:(len(data_input_ls[1][i]))] = data_input_ls[1][i]
            if self.feature_method_2.USE_CMVN == True:
                x2[i] = CMVN(x2[i])

        base_pred = self.get_base_predict(input_=[x,x2])

        ## Cut slice by VAD
        timestamp = self.timestamp(base_pred,empty_num=(self.output_size-1),padding=self.vad_padding)
        base_pred_r=base_pred.reshape(base_pred.shape[0]*base_pred.shape[1],base_pred.shape[2])
        num_slice = len(timestamp)
        max_len = input_len
        if num_slice != 0:
            max_len = max([(t[1]-t[0]) for t in timestamp])
        new = np.zeros((num_slice,max_len,self.output_size))
        new[:,:,-1] = 1
        for idx,t in enumerate(timestamp):
            len_t = t[1]-t[0]
            new[idx,0:len_t] = base_pred_r[t[0]:t[1],:]

        in_len_new = np.zeros((num_slice),dtype=np.int32)
        in_len_new[:] = max_len
        
        ## Decode
        r = K.ctc_decode(new, in_len_new, greedy = True, beam_width=100, top_paths=1)
        r1 = K.get_value(r[0][0]) #(batch,200)
        
        pinyin_sequence = []

        for i in range(num_slice):
            pinyin_sequence.append(
                [[num for num in r1[i] if num != -1],
                list(map(lambda x:x*80,timestamp[i]))]
                ) # exclude -1
        
        return pinyin_sequence #num

    def coldstart(self):
        print('[info] Coldstart')
        self.RecognizeSpeech_FromFile(config.test_audio)
        print('[info] Coldstart ended')

class asr(asr_base): # single_input

    def __init__(self,ver=config.VERSION,**kwargs):
        self.VERSION = ver

        self.MODEL_NAME = f"PINYIN_{self.VERSION}_base"
        self.MODEL_PATH = config.speech_model_folder + self.VERSION + self.slash

        if not config.USE_TFX:
            assert os.path.isdir((self.MODEL_PATH+self.MODEL_NAME))
            self.base_model = load_model((self.MODEL_PATH+self.MODEL_NAME))
        
        self.mode = self.VERSION[-1].upper()
        assert self.mode == 'A'

        self.bg = np.load(config.bg_folder+config.FEATURE_BG)
        self.bg_2 = np.load(config.bg_folder+config.FEATURE_BG_2)

    def Change_to_symbol(self,num):

        return [self.list_symbol_dic_1424[n] for n in num]

    def RecognizeSpeech_FromFile(self, filename):

        try:

            input_length = config.OUTPUT_LENGTH

            wavsignal,fs,num_channel = read_wav_data(filename,True)

            return_list = []
            for i in range(num_channel):

                data_input,_ = self.Get_data_input(wavsignal,fs,i)
                num_slice = self._cal_slice(data_input.shape[0])
                data_input_ls=(self._cut_slice(data_input,num_slice))

                r = self.Predict_single(data_input_ls,input_length)

                for idx in r:
                    return_list.append({"ChannelId":f'{i}',"Text":self.Change_to_symbol(idx[0]),"Time":idx[1]})

            return_list = sorted(return_list, key=lambda i:i['Time'][0])
            
            return return_list
        
        except AudioDecodeError:
            raise AudioDecodeError

        except FileNotFoundError:
            raise FileNotFoundError

class asr_double(asr):

    def RecognizeSpeech_FromFile(self, filename):

        try:

            input_length = config.OUTPUT_LENGTH

            wavsignal,fs,num_channel = read_wav_data(filename,True)

            return_list = []
            for i in range(num_channel):

                data = self.Get_data_input(wavsignal,fs,i)
                num_slice = self._cal_slice(data[0].shape[0])
                data_input_ls=(self._cut_slice(data[0],num_slice),self._cut_slice(data[1],num_slice))

                r = self.Predict_double(data_input_ls,input_length)

                for idx in r:
                    return_list.append({"ChannelId":f'{i}',"Text":self.Change_to_symbol(idx[0]),"Time":idx[1]})

            return_list = sorted(return_list, key=lambda i:i['Time'][0])
            
            return return_list

        except AudioDecodeError:
            raise AudioDecodeError

        except FileNotFoundError:
            raise FileNotFoundError
        
class asr_e2e(asr_base):

    def __init__(self,ver=config.VERSION,**kwargs):
        self.VERSION = ver

        self.MODEL_NAME = f"PINYIN_{self.VERSION}_base"
        self.MODEL_PATH = config.speech_model_folder + self.VERSION + self.slash

        if not config.USE_TFX:
            assert os.path.isdir((self.MODEL_PATH+self.MODEL_NAME))
            self.base_model = load_model((self.MODEL_PATH+self.MODEL_NAME))
        
        self.mode = self.VERSION[-1].upper()
        assert (self.mode == 'C')

        self.bg = np.load(config.bg_folder+config.FEATURE_BG)
        self.bg_2 = np.load(config.bg_folder+config.FEATURE_BG_2)

    def Change_to_word(self,num):

        word_seq = [self.list_word[n] for n in num]

        if len(word_seq) >= 1:
            return (''.join(word_seq))
        else:
            return ''

    def RecognizeSpeech_FromFile(self, filename):

        try:

            input_length = config.OUTPUT_LENGTH

            wavsignal,fs,num_channel = read_wav_data(filename,True)

            return_list = []
            for i in range(num_channel):

                data_input,_ = self.Get_data_input(wavsignal,fs,i)
                num_slice = self._cal_slice(data_input.shape[0])
                data_input_ls=(self._cut_slice(data_input,num_slice))

                r = self.Predict_single(data_input_ls,input_length)

                for idx in r:
                    return_list.append({"ChannelId":f'{i}',"Text":self.Change_to_word(idx[0]),"Time":idx[1]})

            return_list = sorted(return_list, key=lambda i:i['Time'][0])
            
            return return_list
        
        except AudioDecodeError:
            raise AudioDecodeError

        except FileNotFoundError:
            raise FileNotFoundError

class asr_e2e_double(asr_e2e):

    def RecognizeSpeech_FromFile(self, filename):

        try:

            input_length = config.OUTPUT_LENGTH

            wavsignal,fs,num_channel = read_wav_data(filename,True)

            return_list = []
            for i in range(num_channel):

                data = self.Get_data_input(wavsignal,fs,i)
                num_slice = self._cal_slice(data[0].shape[0])
                data_input_ls=(self._cut_slice(data[0],num_slice),self._cut_slice(data[1],num_slice))

                r = self.Predict_double(data_input_ls,input_length)

                for idx in r:
                    return_list.append({"ChannelId":f'{i}',"Text":self.Change_to_word(idx[0]),"Time":idx[1]})

            return_list = sorted(return_list, key=lambda i:i['Time'][0])
            
            return return_list

        except AudioDecodeError:
            raise AudioDecodeError

        except FileNotFoundError:
            raise FileNotFoundError
        