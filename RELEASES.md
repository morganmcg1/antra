# Release History

## v0.2
Better data (HuggingFace nlp paracrawl), cleaned data (via UMAP), larger vocab (+2k for en, + 10k for ga) therefore larger model (74M -> 86M), added SacreBLEU measurement on Tatoeba, 10e training (via 2 x `fit_one_cycle` runs)

#### Data
- Paracrawl en-ga from HuggingFace `nlp` lib, 334k rows

#### Data Processing
- Removed noisy data via UMAP/bokeh visualisation
- lowercase
- Removed samples longer than 60 tokens (90th percentile was 58 tokens long)

#### Tokenizer
- Spacy tokenizer
- [fastai rules](http://dev.fast.ai/text.core#Preprocessing-rules): [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, replace_all_caps, replace_maj, lowercase]
- Vocab size: en :22.9k, ga: 30k

#### Model
- Positional encoding (sin/cos)
- PyTorch nn.Transformer
    - Param count: 86M
    - enc_layers: 6
    - dec_layers: 6
    - n_heads: 8
    - d_model: 512
    - d_inner: 2048
    - vocab size: 22.9k en, 30k ga
    
#### Training
- Fastai: 
    - fit_one_cycle(5, 5e-4, div=5)
    - fit_one_cycle(5, 1e-5, div=5)
    - 15m per epoch
    
#### Performance
- Tatoeba: 25.14 SacreBELU
- CorpusBLEU: 0.503 (20% random validation, random seed = 42)
- Val loss: 0.528, Val Loss: 0.813
- Val Accuracy: 0.613
- Val Perplexity: 2.256

#### Serving
- Added decoding of special tokens for output
- Logging: added inference time logging

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


