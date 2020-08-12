'''
    Functions used to tokenize and de-tokenizer
'''
import csv
from functools import partial
from collections import defaultdict #Counter,namedtuple,OrderedDict
import spacy
from spacy.symbols import ORTH
from .text_helpers import *
from torch import tensor

inp=["WHAT if I can't, what ever shall we do?", "WHAT if I can't, what ever shall we do?"]

def open_vocab(fpath):
    vocab=[]
    with open(fpath, newline='') as csvfile:
        v_reader = csv.reader(csvfile, delimiter=',')
        for row in v_reader: vocab.append(row)
        vocab = [v for sub_v in vocab for v in sub_v]
        return vocab

def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

class SpacyTokenizer():
    "Spacy tokenizer for `lang`"
    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
        for w in self.special_toks: nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe,self.buf_sz = nlp.pipe,buf_sz

    def __call__(self, items):
        return (L(doc).attrgot('text') for doc in self.pipe(map(str,items), batch_size=self.buf_sz))

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
    #def encode(self, o:list): return tensor([self.src_o2i  [o_] for o_ in o])
    def encode(self, o:list): return tensor([self.src_o2i  [o_] for o_ in o])
    def decode(self, o:list): return [self.trg_vocab[o_] for o_ in o] # if self.vocab[o_] != self.pad_tok)