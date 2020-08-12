import streamlit as st
from datetime import date
from os import path
import time

from utils.tokenizer_funcs import spacy_fastai, Numericalize, open_vocab
from utils.processing import fastai_process_trans
from utils.logging import log_usage
from utils.model_utils import load_quantized_model
from utils.translate_utils import translate
from utils.v01_en_ga_transformer import pt_Transformer as ModelClass

config={'model_v':'0.2',
    #'model_path':'models/paracrawl_en_ga_5e_5e-4_5e_1e-5_v0.2_exp4.pth',
    'model_path':'models/paracrawl_en_ga_5e_5e-4_5e_1e-5_v0.2_exp4_no_opt_quantized',
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
    model = load_quantized_model(model_path=model_path, ModelClass=ModelClass, src_vcbsz=len(en_vocab), 
                                    trg_vcbsz=len(ga_vocab), d_model=d_model, d_inner=d_inner)
    
    # STREAMLIT PAGE SETUP
    st.markdown(html_title.format('royalblue','white'),unsafe_allow_html=True)
    st.text('')

    #st.markdown('## 🇬🇧👇')
    st.markdown(inp_html,unsafe_allow_html=True)
    src_txt=st.text_area('', height=50, max_chars=280)

    # TRANSLATE CODE
    if st.button('Translate'):
        trans_start = time.time()
        st.text('')
        st.text('')
        trg_txt=translate(src_txt=src_txt,model=model,tokenizer=tokenizer,numericalizer=numericalizer)
        trg_txt=fastai_process_trans(trans=trg_txt)[0]

        st.markdown(out_html,unsafe_allow_html=True)
        st.markdown(f"## \n \
        > {trg_txt}")
        trans_end = time.time()
        inf_time = trans_end - trans_start
        log_usage(src_txt,trg_txt,inf_time=inf_time,feedback=None,model_v=model_v)
    
    # see html from here for layout ideas: https://discuss.streamlit.io/t/st-button-in-a-custom-layout/2187/2

    # SIDEBAR CODE
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

if __name__ == '__main__':
	main()