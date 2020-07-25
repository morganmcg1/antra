import streamlit as st
import torch
from einops import rearrange
from fastai2.basics import *
import pandas as pd
import math
from datetime import date,datetime
import pytz
import csv
import os.path
from os import path
import time

from tokenizers import spacy_fastai, Numericalize
from v01_en_ga_transformer import pt_Transformer as ModelClass

# Tokenize
#src_text=["WHAT if I can't, what ever shall we do?", "WHAT if I can't, what ever shall we do?"]    

config={'model_v':'0.2',
    'model_path':'models/paracrawl_en_ga_5e_5e-4_5e_1e-5_v0.2_exp4.pth',
    'd_model':512,
    'd_inner':2048,
    'en_vocab_path':'data/paracrawl_vocab_en_v0.2_exp4.csv',
    'ga_vocab_path':'data/paracrawl_vocab_ga_v0.2_exp4.csv'
    }

model_v=config['model_v']
model_path=config['model_path']
d_model=config['d_model']
d_inner=config['d_inner'] 
en_vocab_path=config['en_vocab_path']
ga_vocab_path=config['ga_vocab_path']

# STYLING
html_title = """
    <div style="background-color:{};padding:10px;border-radius:10px">
    <h1 style="color:{};text-align:center;">AnTrá </h1>
    </div>
    """
inp_html = 	"""<h2 style=text-align:center;">🇬🇧👇 </h2>"""
out_html = 	"""<h2 style=text-align:center;">🇮🇪👇 </h2>"""

# RUN APP
def main():
    # TOKENIZER SETUP 
    en_vocab=open_vocab(en_vocab_path)
    ga_vocab=open_vocab(ga_vocab_path)

    tokenizer=spacy_fastai()
    numericalizer=Numericalize(en_vocab, ga_vocab)
    
    # LOAD MODEL

    start = time.time()
    model = load_model(model_path=model_path, ModelClass=ModelClass, src_vcbsz=len(en_vocab), trg_vcbsz=len(ga_vocab),
                        d_model=d_model, d_inner=d_inner)

    # STREAMLIT SETUP
    st.markdown(html_title.format('royalblue','white'),unsafe_allow_html=True)
    st.text('')

    #st.markdown('## 🇬🇧👇')
    st.markdown(inp_html,unsafe_allow_html=True)
    src_txt=st.text_area('', height=50, max_chars=280)

    # TRANSLATE
    if st.button('Translate'):
        trans_start = time.time()
        st.text('')
        st.text('')
        #with st.spinner('Wait for it...'):
        trg_txt=translate(src_txt=src_txt,model=model,tokenizer=tokenizer,numericalizer=numericalizer)
        trg_txt=fastai_process_trans(trans=trg_txt)[0]
        #st.success('Done!')

        st.markdown(out_html,unsafe_allow_html=True)
        st.markdown(f"## \n \
        > {trg_txt}")
        trans_end = time.time()
        inf_time = trans_end - trans_start
        log_usage(src_txt,trg_txt,inf_time=inf_time,feedback=None,model_v=model_v)
    
    # see html from here for layout ideas: https://discuss.streamlit.io/t/st-button-in-a-custom-layout/2187/2

    # SIDEBAR
    st.sidebar.markdown('## About \n AnTrá is an Irish Language Toolset \n \n Translate from Enlglish to Irish \
    and copy your translation to wherever you need to paste it')
    st.sidebar.markdown("---")
    st.sidebar.text('')
    if st.sidebar.checkbox('Show Release Notes'):
        st.sidebar.markdown(f'This is v{model_v}, \n [see here](https://github.com/morganmcg1/antra/blob/master/RELEASES.md)\
            for full release notes')

    # FORMATTING
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def log_usage(src_txt, trg_txt,feedback=None,model_v=None,inf_time=None):
    log_fn='usage_logs2.csv'
    feedback='na'
    naive_dt = datetime.now()
    tz='Europe/Dublin'
    indy = pytz.timezone(tz)
    dt = indy.localize(naive_dt)

    fields=[dt,tz,src_txt,trg_txt,feedback,inf_time]

    if path.exists(log_fn):
        with open(log_fn, 'a') as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(fields)
    else:
         with open(log_fn, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(fields)

# TRANSLATION CODE
def open_vocab(fpath):
    en_vocab=pd.read_csv(fpath,header=None)
    en_vocab.columns=['col1']
    return en_vocab.col1.values

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

def fastai_process_trans(trans):
    trans_ls=[]
    for s in trans: 
        #print(s)
        tmp = s.replace('xxbos','')
        tmp = tmp.replace('xxeos','')
        tmp = tmp.replace(' .','.')
        tmp = tmp.replace(' ,',',')
        tmp = tmp.replace(' ?','?')
        tmp = tmp.replace(' !','!')
        #print(tmp[0])
        if tmp.endswith('. '): tmp=tmp[:-1]
        if tmp.endswith('? '): tmp=tmp[:-1]
        if tmp.endswith('! '): tmp=tmp[:-1]

        for spec in ['xxmaj ', 'xxup ']:
            found=[]
            for m in re.finditer(spec, tmp):
                found.append(m.start())
            for f in found:
                m = tmp.find(spec)
                if m != -1:   
                    ml = m+len(spec)
                    tmp = tmp[:ml] + tmp[ml].upper() + tmp[ml+1:]
                    if m != 0:
                        tmp = tmp[:m] + tmp[ml:]
                    else: 
                        tmp = tmp[ml:]

        found=[]    
        xxwrep = 'xxwrep '            
        for m in re.finditer(xxwrep, tmp):
            found.append(m.start())
        for f in found:
            m = tmp.find(xxwrep)
            n = int(tmp[m+7])    # number of repetitions of word
            pwrep = m+8    # position where repeated word starts
            wrep = tmp[pwrep:].split()[0]    # word to be repeated
            lwrep = len(wrep)    # length of repeated word
            tmp = tmp[:m] + f"{wrep} " * n + tmp[pwrep+lwrep+1:]
        
        # Remove space at start
        if tmp[0] == ' ': tmp = tmp[1:]            
        trans_ls.append(tmp)
    return trans_ls

@st.cache()
def load_model(model_path=None, ModelClass=None, src_vcbsz=None, trg_vcbsz=None, d_model=None, d_inner=None):
    model = ModelClass(src_vcbsz=src_vcbsz, trg_vcbsz=trg_vcbsz, d_model=d_model, d_inner=d_inner)
    state=torch.load(model_path, map_location=torch.device('cpu'))
    model_state = state['model']
    model.load_state_dict(model_state)
    #model.reset()
    return model.eval()

def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

@st.cache()
def forward_model(src_toks:list=None, trg_toks:list=None, model=None):
    src = torch.as_tensor(src_toks).unsqueeze(0).long()
    tgt = torch.as_tensor(trg_toks).unsqueeze(0)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    with torch.no_grad():
        output = model.forward(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, 
                                tgt_key_padding_mask=None, memory_key_padding_mask=None)

    # fastai inference (from Zach)
    # with torch.no_grad():
    #     learn.model.reset()
    #     learn.model.eval()
    #     out = learn.model(*batch)
    # learn.loss_func.decodes(out[0])

    return output.squeeze(0).detach()

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

if __name__ == '__main__':
	main()