## Introduction

&nbsp; Tokenization plays a crucial role in Natural Language Processing. However, comparative studies focused on tokenization are rare. This is especially true in Natural Language Generation process based on small data and model structures.
To address this issue, this repository covers comparative analysis of the impact of four distinct Tokenization approaches on the performance of Neural Machine Translation task. By doing so, I hope to establish a valuable benchmark that can serve as a reference point for future research endeavors.

<br><br>

## Background

**Word Tokenization**
  * Split Text based on Words according to separator like white space
  * The most simple and intuitive Tokenization Methodology
  * Large vocab size is essential for various expressive abilities
  * Easily get into Out of Vocabulary trouble, and has difficulty to respond to new words
<br>

**Character Tokenization**
  * Split Text based on Character
  * Only vocab size equal to the number of alphabets appearing in the text is required
  * Out of Vocabulary rarely occurs, and easy to respond to new words
  * difficult to train model because the model has to learn a lot of token combinations
<br>

**Sub-Word Tokenization**
  * Intermediate form of Word Toknenization and Character Toknenization
  * Split text based on subwords, which are smaller than words yet larger than characters
  * There are various algorithms for how to construct sub words
  * Possible to flexibly cope with new expressions, and can prevent token expressions getting too long
  * most commonly used on various models

</br></br>


## Tokenizers

**Word-Level Tokenizer** <br>
> This is the “classic” tokenization algorithm. It let’s you simply map words to IDs without anything fancy. This has the advantage of being really simple to use and understand, but it requires extremely large vocabularies for a good coverage. Using this Model requires the use of a PreTokenizer. No choice will be made by this model directly, it simply maps input tokens to IDs.

<br>


**Word Piece Tokenizer** <br>
> This is a subword tokenization algorithm quite similar to BPE, used mainly by Google in models like BERT. It uses a greedy algorithm, that tries to build long words first, splitting in multiple tokens when entire words don’t exist in the vocabulary. This is different from BPE that starts from characters, building bigger tokens as possible. It uses the famous ## prefix to identify tokens that are part of a word (ie not starting a word).

<br>

**Byte Pair Encoding Tokenizer** <br>
> One of the most popular subword tokenization algorithm. The Byte-Pair-Encoding works by starting with characters, while merging those that are the most frequently seen together, thus creating new tokens. It then works iteratively to build new tokens out of the most frequent pairs it sees in a corpus. BPE is able to build words it has never seen by using multiple subword tokens, and thus requires smaller vocabularies, with less chances of having “unk” (unknown) tokens.

<br>

**Unigram** <br>
> Unigram is also a subword tokenization algorithm, and works by trying to identify the best set of subword tokens to maximize the probability for a given sentence. This is different from BPE in the way that this is not deterministic based on a set of rules applied sequentially. Instead Unigram will be able to compute multiple ways of tokenizing, while choosing the most probable one.

</br></br>


## Experimental Setups

| Model Setup | Training Setup |
|---|---|
| &nbsp; **`Architecture:`** &nbsp; Transformer &emsp; | &nbsp; **`N_Epochs:`** &nbsp; 10 |
| &nbsp; **`Embedding Dimension:`** &nbsp; 256 &emsp;  | &nbsp; **`Batch Size:`** &nbsp; 32 |
| &nbsp; **`Hidden Dimension:`** &nbsp; 256            | &nbsp; **`LR:`** &nbsp; 5e-4 |
| &nbsp; **`FFN Dimension:`** &nbsp; 512               | &nbsp; **`iters_to_accumulate:`** &nbsp; 4 &emsp; |
| &nbsp; **`N Heads:`** &nbsp; 8                       | &nbsp; **`Gradient Clip Max Norm:`** &nbsp; 1 &emsp; |
| &nbsp; **`N Layers:`** &nbsp; 3                      | &nbsp; **`Apply AMP:`** &nbsp; True |

</br></br>

## Evaluation

| &emsp; Tokenizer Type &emsp; | &emsp; 10k Model Score &emsp; | &emsp; 20k Model Score &emsp; | &emsp; 30k Model Score &emsp; |
|:---:|:---:|:---:|:---:|
| **`Word Level`** | 16.09 | 12.80 | 12.50 |
| **`Word Piece`** | 19.96 | 14.62 | 12.17 |
| **`BPE`**        | 13.32 | 13.39 | 13.28 |
| **`Unigram`**    | 13.58 | 13.88 | 15.64 |

<br><br> 


## How to Use
```
git clone https://github.com/moon23k/Tokenizers.git
```

```
cd Tokenizers
python3 setup.py -tokenizer_type ['all', 'WL', 'WP', 'BPE', 'UNI']
                 -vocab_size [10k, 20k, 30k]
python3 run.py -mode [train, test, inference]
               -tokenizer_type ['WL', 'WP', 'BPE', 'UNI']
               -vocab_size [10k, 20k, 30k]
```
<br><br> 

## Reference
* [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)
<br> 
