'''
    Functions used to load and call the model
'''

import torch
from einops import rearrange
import streamlit as st

# ------------------- LOADING MODEL ------------------- 
@st.cache()
def load_model(model_path=None, ModelClass=None, src_vcbsz=None, trg_vcbsz=None, d_model=None, d_inner=None):
    model = ModelClass(src_vcbsz=src_vcbsz, trg_vcbsz=trg_vcbsz, d_model=d_model, d_inner=d_inner)
    state=torch.load(model_path, map_location=torch.device('cpu'))
    model_state = state['model']
    model.load_state_dict(model_state)
    #model.reset()
    return model.eval()

@st.cache()
def load_quantized_model(model_path=None, ModelClass=None, src_vcbsz=None, trg_vcbsz=None, d_model=None, d_inner=None):
    model = ModelClass(src_vcbsz=src_vcbsz, trg_vcbsz=trg_vcbsz, d_model=d_model, d_inner=d_inner)
    # Quantize model definition
    quantized_model = torch.quantization.quantize_dynamic(model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)
    # Load state dict
    state_dict=torch.load(model_path, map_location=torch.device('cpu'))
    # Load state dict into model
    quantized_model.load_state_dict(state_dict)
    #model.reset()
    return quantized_model.eval()

 #------------------- CALLING MODEL ------------------- 
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
    return output.squeeze(0).detach()   