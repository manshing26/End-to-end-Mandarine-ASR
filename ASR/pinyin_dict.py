import config

from pypinyin import pinyin, lazy_pinyin, Style

class Pinyin_dict:
    '''
    Change str(word_sentence) -> List(pinyin)
    '''

    def __init__(self):
        self.build_dict()

    def build_dict(self):
        with open(config.word_dict,'r')as f:
            data = f.read()
            self.word_ls = data.split('\n')[:-1]
        self.pinyin_dict = {}
        
        for w in self.word_ls:
            self.pinyin_dict[w] = []

        with open(config.pinyin_dict,'r')as f:
            data = f.read()
            line = data.split('\n')[:-1]
        del data
        element = [l.split('\t') for l in line]

        for e in element:
            for ww in e[1]:
                if ww in self.pinyin_dict.keys():
                    self.pinyin_dict[ww].append(e[0])

    def _check_self_dict(self,string=''): # self.pinyin list
        r = []
        if len(string) == 0:
            return r
        else:
            for char in string:
                r.append(self.pinyin_dict.get(char))
        return r

    def _check_lazy_pinyin(self,string=''):
        r = []
        if len(string) == 0:
            return r
        else:
            r1 = lazy_pinyin(string,Style.TONE3)
            return r1

    def _further_match_2(self,char): # for one char
        origin = self._check_self_dict(char)
        from_lib = pinyin(char,Style.TONE3,heteronym=True)
        pass

    @classmethod
    def _further_match(self,lazy,char):
        if len(char) == 0:
            return None
        char_no_tone = [c[:-1] for c in char]
        for idx, c in enumerate(char_no_tone):
            if lazy == c:
                return char[idx]
        return char[0]

    def format_check(self,string):
        try:
            r = self.check(string)
            print("success")
            return r
        except Exception as e:
            print(e)

    def check(self,string:str):

        assert type(string)==str
        r = []
        origrin = self._check_self_dict(string)
        lazy = self._check_lazy_pinyin(string)

        if(len(origrin)!=len(lazy)):
            return [None]

        for idx,char in enumerate(origrin):
            if char == None: # handle none
                r.append(None)
            elif lazy[idx] in char:
                r.append(lazy[idx])
            elif lazy[idx] not in char:
                r.append(self._further_match(lazy[idx],char))
        return r
