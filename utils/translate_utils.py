'''
    Functions used to coordinate translation
'''

import streamlit as st
import torch 
from .model_utils import forward_model

@st.cache()
def translate(src_txt:str=None, model=None, tokenizer=None, numericalizer=None,
                bos_idx=2, eos_idx=3, max_trg_len=45):
    src_toks = tokenizer.tokenize(src_txt)
    src_toks = numericalizer.encode(src_toks)
    trg_toks=[bos_idx]
    trg_toks=translate_seq(src_toks,trg_toks,model,eos_idx=eos_idx)
    trg_toks = clean_trg_toks(trg_toks, bos_idx, eos_idx)
    trg_toks=numericalizer.decode(trg_toks)
    return [' '.join(trg_toks)]

@st.cache()
def translate_seq(src_toks:list=None, trg_toks:list=None, model=None, eos_idx:int=3, max_trg_len:int=45):
    i = 0
    while i < max_trg_len and int(trg_toks[-1]) != eos_idx:
        output = forward_model(src_toks, trg_toks, model)
        values, indices = torch.topk(output, 5)
        trg_toks.append(int(indices[-1][0]))
        i+=1
    return trg_toks

def clean_trg_toks(trg_toks:str=None, bos_idx=2, eos_idx=3):
    trg_toks = [t for t in trg_toks if t != bos_idx] 
    return [t for t in trg_toks if t != eos_idx] 