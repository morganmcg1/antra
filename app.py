import streamlit as st
import torch
from einops import rearrange
from fastai2.basics import *
from tokenizers import spacy_fastai, Numericalize

# st.title('anTrá')
# st.text('')

# src_txt=st.text_area("what shall we translate?", height=50, max_chars=280)
# st.text('')
# st.text('')

def st_tokenize(text, tokenizer):
    return tokenizer.tokenize(text)

# def st_numericalize(tokens, vocab):
    
#     o2i = None if vocab is None else defaultdict(int, {v:k for k,v in enumerate(vocab)})
#     def encodes(self, o): return tensor([o2i  [o_] for o_ in o])
#     def decodes(self, o): return L(self.vocab[o_] for o_ in o if self.vocab[o_] != self.pad_tok)
#     return [o for o in tokens]



def main():
    # MODEL
    model = torch.load('models/paracrawl_en_ga_5e_5e-4.pth', map_location=torch.device('cpu'))
    if model is not None: print('success')

    # Setup 
    tokenizer=spacy_fastai()
    numericalizer=Numericalize(vocab)

    # Tokenize
    src_text=["WHAT if I can't, what ever shall we do?", "WHAT if I can't, what ever shall we do?"]    
    tokens = st_tokenize(src_text[0], tokenizer)
    nums = numericalize.encodes(tokens)
    print(nums)

#print(tokenizer.tokenize(inp[0]))


#learner=
# learner=load_learner('models/paracrawl_en_ga_5e_5e-4_learner.pkl')
# if learner is not None:
#     st.success('its here!')

# # LOAD FUNCS
# def setup_learner():
#     model_path='XXX.pth'
#     learner_path='XXX.pth'
#     #model=load_model(path)
#     learner = learner_path('YYY.pth')
#     model = learner.model
#     model.eval

#     tokenizer = learner.dls.tokenizer[0][1].encodes
#     numericalize = learner.dls.numericalize[0].encodes

#     detokenizer = learner.dls.tokenizer[1][1].decodes
#     denumericalize = learner.dls.numericalize[1].decodes

# # TRANSLATION CODE
# @st.cache()
# def load_model(path:str=None):
#     model = load(path)
#     model.eval()
#     return model

# def gen_nopeek_mask(length):
#     mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask

# def forward_model(src_toks:list=None, trg_toks:list=None, model=None):
#     src = torch.as_tensor(src_toks).unsqueeze(0).long()
#     tgt = torch.as_tensor(trg_toks).unsqueeze(0)
#     tgt_mask = gen_nopeek_mask(tgt.shape[1])
#     output = model.forward(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, 
#                             tgt_key_padding_mask=None, memory_key_padding_mask=None)
#     return output.squeeze(0).detach()

# @st.cache()
# def translate_seq(src_toks:list=None, trg_toks:list=None, model=None, eos_idx:int=3, max_trg_len:int=45):
#     i = 0
#     while i < max_trg_len and int(trg_toks[-1]) != eos_idx:
#         output = forward_model(model, src_toks, trg_toks)
#         values, indices = torch.topk(output, 5)
#         trg_toks.append(int(indices[-1][0]))
#         i+=1
#     return trg_toks

# @st.cache()
# def tokenize(text:str=None, tokenizer=None):
#     return tokenizer(text)

# @st.cache()
# def numericalize(toks:list=None, numericalizer=None):
#     return numericalizer(toks)

# @st.cache()
# def detokenize(toks=None, detokenizer=None):
#     return detokenizer(toks)

# @st.cache()
# def denumericalize(toks:list=None, denumericalizer=None):
#     return denumericalizer(toks)

# def clean_trg_toks(trg_toks:str=None, bos_idx=2, eos_idx=3):
#     trg_toks = [t for t in trg_toks if t != bos_idx] 
#     return [t for t in trg_toks if t != eos_idx] 

# def translate(src_txt:str=None, model=None, tokenizer=None, bos_idx=2, eos_idx=3, max_trg_len=45):
#     src_toks=tokenize(src_txt, tokenizer=None)
#     trg_toks=[bos_idx]
#     trg_toks = translate_seq(trg_toks, model, eos_idx)
#     trg_toks = clean_trg_toks(trg_toks, bos_idx, eos_idx)
#     trg_text = denumericalize(trg_toks, denumericalizer)
#     trg_text = detokenize(trg_text, detokenizer)
#     return trg_text


# # STREAMLIT CODE

# st.markdown('*Perfectly translated output*')
# #trg_txt=st.write(translate(src_txt, model, tokenizer))

# hide_streamlit_style = """
#             <style>
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if __name__ == '__main__':
	main()