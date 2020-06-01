# Release History

## v0.1
Baseline release to be improved upon

#### Data
- Paracrawl en-ga

#### Data Processing
- lowercase
- Removed samples longer than 60 tokens (90th percentile was 58 tokens long)

#### Tokenizer
- Spacy tokenizer
- [fastai rules](http://dev.fast.ai/text.core#Preprocessing-rules): [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, replace_all_caps, replace_maj, lowercase]

#### Model
- PyTorch nn.Transformer
    - Param count: 74M
    - enc_layers: 6
    - dec_layers: 6
    - n_heads: 8
    - d_model: 512
    - d_inner: 2048
    - vocab size: 20k en, 20k ga
    
#### Training
- CorpusBLEU: 0.468 (20% random validation)
- Fastai: fit_one_cycle(20, 5e-4, div=5), 23min per epoch
- Train loss: 0.389287, Val Loss: 0.942813

#### Serving
- Streamlit app
- logging: input, output, datetime, feedback (null), model version

