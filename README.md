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


## Setups

| Small Model Setup | Big Model Setup | Training Setup |
|---|---|---|
| Embedding Dimension: 256 | Embedding Dimension: 512 | N_Epochs: 10 |
| Hidden Dimension: 256 | Hidden Dimension: 512 | LR: 5e-4 |
| FFN Dimension: 512 | FFN Dimension: 1024 | iters_to_accumulate: 4 |
| N Heads: 8 | N Heads: 8 | Gradient Clip Max Norm: 1 |
| N Layers: 3 | N Layers: 3 | Apply AMP: True |


</br></br>

## Results

> **Small Model**

| &emsp; Tokenizer Type &emsp; | &emsp; Vocab Size &emsp; | &emsp; Greedy Score &emsp; | &emsp; Beam Score &emsp; |
|:---:|:---:|:---:|:---:|
| Word Level |  5k | 15.36 | 11.94 |
| -          | 15k | 14.39 | 11.49 |
| -          | 30k | 13.06 | 11.42 |
| Word Piece |  5k | **22.72** | **21.50** |
| -          | 15k | 14.42 | 13.75 |
| -          | 30k | 13.31 | 10.53 |
| BPE        |  5k | 12.83 | 11.84 |
| -          | 15k | 13.78 | 10.53 |
| -          | 30k | 14.63 | 11.00 |
| Unigram    |  5k | **11.33** | 11.05 |
| -          | 15k | 11.44 |  **9.08** |
| -          | 30k | 14.23 | 11.16 |

<br>

In experiments based on the Small model, the performance of Word Level and Word Piece decreases as the vocab size increases, while the performance of BPE and Unigram tends to improve as the vocab size increases. Under the same conditions, the Word Piece method with vocab as much as 5k showed the best performance, and there is not a large deviation from the other methods. 
The small performance deviation can be attributed to the fact that the model size is small and relatively insensitive.


<br><br>
> **Big Model**

| &emsp; Tokenizer Type &emsp; | &emsp; Vocab Size &emsp; | &emsp; Greedy Score &emsp; | &emsp; Beam Score &emsp; |
|:---:|:---:|:---:|:---:|
| Word Level |  5k | 14.05 | 11.90 |
| -          | 15k | 13.82 | 12.52 |
| -          | 30k |  8.73 |  7.41 |
| Word Piece |  5k | **27.75** | **23.61** |
| -          | 15k | 19.13 | 14.35 |
| -          | 30k |  **0.00** |  **0.00** |
| BPE        |  5k | 16.20 | 11.44 |
| -          | 15k |  8.88 |  6.33 |
| -          | 30k | 14.29 |  9.65 |
| Unigram    |  5k | 17.06 | 15.79 |
| -          | 15k | 13.06 | 10.09 |
| -          | 30k |  0.14 |  0.05 |

<br>

In the experiments conducted on the Big model, the 5k-sized WP also showed the best performance. And it is also possible to confirm that the tendency found in the previous Small Model-based experiment is maintained to some extent. However, there is a large variation in performance by vocab size, which seems to be the main reason that the size of the model increases and becomes more sensitive.

</br></br>

## How to Use
```
git clone https://github.com/moon23k/Tokenizers.git
```

```
cd Tokenizers
python3 setup.py
python3 run.py
```
</br>

