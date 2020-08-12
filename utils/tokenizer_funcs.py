'''
    Functions used to tokenize and de-tokenizer
'''
import csv
from fastai2.basics import *
from fastai2.text.core import defaults, lowercase, SpacyTokenizer 

inp=["WHAT if I can't, what ever shall we do?", "WHAT if I can't, what ever shall we do?"]

def open_vocab(fpath):
    vocab=[]
    with open(fpath, newline='') as csvfile:
        v_reader = csv.reader(csvfile, delimiter=',')
        for row in v_reader: vocab.append(row)
        vocab = [v for sub_v in vocab for v in sub_v]
        return vocab

def maps(*args, retain=noop):
    "Like `map`, except funcs are composed first"
    f = compose(*args[:-1])
    def _f(b): return retain(f(b), b)
    return map(_f, args[-1])

def compose(*funcs):
    "Modifed from fastcore library"
    "Create a function that composes all functions in `funcs`, passing along remaining `*args` and `**kwargs` to all"
    if len(funcs)==0: return noop
    if len(funcs)==1: return funcs[0]
    def _inner(x, *args, **kwargs):
        for f in funcs: x = f(x, *args, **kwargs)
        return x
    return _inner

class spacy_fastai():
    def __init__(self, tok=SpacyTokenizer(), defaults=defaults, lowercase=lowercase): 
        self.tok = tok
        self.proc_rules = defaults.text_proc_rules[:-1].copy() + [partial(lowercase, add_eos=True)]
        self.post_f = compose(defaults.text_postproc_rules)
        self.defaults=defaults

    def tokenize(self, text:str):
        assert type(text)==str, "tokenize must receive a string"
        "Takes a string, returns a list of tokens"
        text=text.lower()
        pre_processed=compose(*self.proc_rules)(text)
        tokens = list(self.tok([pre_processed]))
        post_processed = list(list(maps(*self.defaults.text_postproc_rules, o)) for o in tokens)
        return [o for sl in post_processed for o in sl]

class Numericalize():
    def __init__(self, src_vocab:list=None, trg_vocab:list=None):
        self.src_vocab,self.trg_vocab = src_vocab,trg_vocab
        self.src_o2i = None if src_vocab is None else defaultdict(int, {v:k for k,v in enumerate(src_vocab)})
        #self.trg_o2i = None if trg_vocab is None else defaultdict(int, {v:k for k,v in enumerate(trg_vocab)})
    def encode(self, o:list): return tensor([self.src_o2i  [o_] for o_ in o])
    def decode(self, o:list): return [self.trg_vocab[o_] for o_ in o] # if self.vocab[o_] != self.pad_tok)

