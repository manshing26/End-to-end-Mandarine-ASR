'''
Calculate the performance of trained models
'''
import config
from ASR.readdata_json import DataSpeech, DataSpeech2
from ASR.readdata_combine import DataSpeech as DataSpeechC
from ASR.asr import asr, asr_e2e
from ASR.file_function.logic_func import *
from ASR.file_function.file_func import *

import click
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import backend as K


class performance_cal:

    def __init__(self,ver_list,batch_size=5,dataset='test',go_backward=False,**kwargs):
        '''
        Mode A
        '''
        self.b_size = batch_size
        self.datapath = config.data_path
        self.ver_list = ver_list
        self.num_model = len(self.ver_list)
        self.score = [0]*self.num_model
        self.mode = [None]*self.num_model
        self.total_sample=0
        self.ver_dict = {}
        for v in self.ver_list:
            self.ver_dict[v] = asr(ver=v,mode='dev')

        if kwargs['yyihu']:
            self.data_class = DataSpeechC(path=self.datapath,type_=dataset,preload=False)
        else:
            self.data_class = DataSpeech(path=self.datapath,type_=dataset)

        if config.DOUBLE_INPUT:
            self.data_gen = self.data_class.data_generator_double(batch_size=self.b_size,go_backward=go_backward)
        else:
            self.data_gen = self.data_class.data_generator_single(batch_size=self.b_size,go_backward=go_backward)

    def run(self,run_no=25):
        for r in tqdm(range(run_no)):
            if config.DOUBLE_INPUT == True:
                self.cal_double() # A:v C:x
            elif config.DOUBLE_INPUT == False:
                self.cal_single() # A:x C:v
        print('\n[result]: Word Error Rate\n')
        for i in range(self.num_model):
            print(f'{self.ver_list[i]} : {self.score[i]/self.total_sample}\n')
        print('\n[result]: Accuracy\n')
        for i in range(self.num_model):
            print(f'{self.ver_list[i]} : {(1-(self.score[i]/self.total_sample))*100:.3f}%\n')

    def cal_double(self): # Double input
        data = next(self.data_gen)[0]
        x_batch = data[0]
        x2_batch = data[1]
        y_batch = data[2]
        in_len = data[3]
        label_len = data[4]

        for i in range(self.num_model):

            result = self.ver_dict[self.ver_list[i]].base_model.predict(x=[x_batch,x2_batch])

            for b in range(self.b_size):

                shape = result[b].shape
                temp = result[b].reshape(1,shape[0],shape[1])
                
                try:
                    r = K.ctc_decode(temp,in_len[b],greedy=True,beam_width=100,top_paths=1)
                except Exception:
                    print('[error] ctc_decoding error')
                    continue 

                r = K.get_value(r[0][0])[0]

                y_pred = [str(num) for num in r if num != -1]
                y_true = y_batch[b,0:int(label_len[b])]
                y_true = [str(y) for y in y_true]

                self.score[i] += WordErrorRate(y_true,y_pred)

        self.total_sample += self.b_size

    def cal_single(self): # single input
        data = next(self.data_gen)[0]
        x_batch = data[0]
        y_batch = data[1]
        in_len = data[2]
        label_len = data[3]

        for i in range(self.num_model):

            result = self.ver_dict[self.ver_list[i]].base_model.predict(x=x_batch)

            for b in range(result.shape[0]): #self.batch_size

                shape = result[b].shape
                temp = result[b].reshape(1,shape[0],shape[1])
                
                try:
                    r = K.ctc_decode(temp,in_len[b],greedy=True,beam_width=100,top_paths=1)
                except Exception:
                    print('[error] ctc_decoding error')
                    continue 

                r = K.get_value(r[0][0])[0]

                y_pred = [str(num) for num in r if num != -1]
                y_true = y_batch[b,0:int(label_len[b])]
                y_true = [str(y) for y in y_true]

                self.score[i] += WordErrorRate(y_true,y_pred)

        self.total_sample += self.b_size

class performance_cal_C(performance_cal):

    def __init__(self,ver_list,batch_size=5,dataset='test',go_backward=False,**kwargs):
        '''
        Mode C ; for e2e model
        '''
        self.b_size = batch_size
        self.datapath = config.data_path
        self.ver_list = ver_list
        self.num_model = len(self.ver_list)
        self.score = [0]*self.num_model
        self.mode = [None]*self.num_model
        self.total_sample=0
        self.ver_dict = {}
        for v in self.ver_list:
            self.ver_dict[v] = asr_e2e(ver=v)

        if kwargs['yyihu']:
            self.data_class = DataSpeechC(path=self.datapath,type_=dataset,preload=False)
        else:
            self.data_class = DataSpeech2(path=self.datapath,type_=dataset)
            
        if config.DOUBLE_INPUT:
            self.data_gen = self.data_class.data_generator_double(batch_size=self.b_size,go_backward=go_backward)
        else:
            self.data_gen = self.data_class.data_generator_single(batch_size=self.b_size,go_backward=go_backward)

@click.command()
@click.option('-m','--mode',help="Mode of dataset [train/dev/test]",type=str,default='test',show_default=True)
@click.option('-e','--epoch',help="Epcho of performance testing",type=int,default=200,show_default=True)
@click.option('-y','--yyihu',help="Use yyihu data as half",type=bool,default=False,show_default=True)
@click.argument('version', nargs=-1, type=str)
def main(mode:str,epoch:int,version,yyihu:bool):

    assert not config.USE_TFX, 'Using tensorflow serving'

    if len(version) != 0:
        ver_list = list(version)
    else:
        ver_list = [config.VERSION]

    mode_list = [v[-1].lower() for v in ver_list]

    if (('c' in mode_list) and ('a' in mode_list)):
        raise Exception('Mode A and mode C are not compatible')

    dataset_ls = ['train','dev','test']
    if (mode.lower() in dataset_ls):
        dataset = mode.lower()

    if 'c' in mode_list:
        pc = performance_cal_C(ver_list,dataset=dataset,go_backward=False,yyihu=yyihu)
    elif 'a' in mode_list:
        pc = performance_cal(ver_list,dataset=dataset,go_backward=False,yyihu=yyihu)
    else:
        exit()
        
    pc.run(epoch)
    K.clear_session()

if __name__=="__main__":

    gpu = config.GPU

    if gpu:
        configP = tf.compat.v1.ConfigProto()
        configP.gpu_options.allow_growth=True
        sess = tf.compat.v1.Session(config=configP)

    main()