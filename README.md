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
| &emsp; **`Embedding Dimension:`** &nbsp; 256 &emsp; | &emsp; **`N_Epochs:`** &nbsp; 10 |
| &emsp; **`Hidden Dimension:`** &nbsp; 256           | &emsp; **`LR:`** &nbsp; 5e-4 |
| &emsp; **`FFN Dimension:`** &nbsp; 512              | &emsp; **`iters_to_accumulate:`** &nbsp; 4 &emsp; |
| &emsp; **`N Heads:`** &nbsp; 8                      | &emsp; **`Gradient Clip Max Norm:`** &nbsp; 1 &emsp; |
| &emsp; **`N Layers:`** &nbsp; 3                     | &emsp; **`Apply AMP:`** &nbsp; True |

</br></br>

## BLEU Evaluation Test

| &emsp; Tokenizer Type &emsp; | &emsp; 10k Model Score &emsp; | &emsp; 20k Model Score &emsp; | &emsp; 30k Model Score &emsp; |
|:---:|:---:|:---:|:---:|
| **`Word Level`** | 16.45 | 15.74 | 14.89 |
| **`Word Piece`** | 23.04 | 14.09 | 14.46 |
| **`BPE`**        | 14.22 | 14.90 | 13.23 |
| **`Unigram`**    | 14.16 | 15.47 | 14.77 |

<br><br>

## Generation Test
<br>

> **Word-Level Tokenization Model**

| Vocab_size | Sequence_Type | Sequence |
| :---: | :---: | :--- |
| **`10k`**  | Generated        | nun sind die vorschriften , die , die , bei denen und fur verwendet werden . |
| -          | Back Translation |  now the regulations are used for which , which , where and for . |
| **`20k`**  | Generated        | es gibt vorschriften , die nach , die dann , wie aus substanzen in der industrie verwendet werden .|
| -          | Back Translation | There are regulations that are made according to how substances are used in industry. |
| **`30k`**  | Generated        | jetzt sind regelungen zur , die sich fur hinaus , die , als fur dieser sektor eingesetzt werden . |
| -          | Back Translation | There are now regulations in place that are intended to be used in this sector. |

<br><br>

> **Word-Piece Tokenization Model**

| Vocab_size | Sequence_Type | Sequence |
| :---: | :---: | :--- |
| **`10k`**  | Generated        | es gibt regelungen , die die mit bio ##met ##ho ##chs ##t ##grenzen , die mit der freis ##etzung von pfl ##anzen und energie fur die industrie verwendet ##en menge ##n . |
| -          | Back Translation | there are regulations that limit the bio ##met ##ho ##chs ##t ##t ##t ##t ##s with the release of plants ## and energy used for the industry ## amount ##n . |
| **`20k`**  | Generated        | jetzt , was die quantitative ##n organis ##che ##idung ##en fur die organis ##ierung von brenn ##stoffe , die mit brenn ##stoffe und auch fur chem ##ische anlagen in die industrie verwendeten art verwendet werden . |
| -          | Back Translation | Now what the quantitative ##n organ ##che ##idung ##en for the organization ##ing of fuel ##materials used with fuel ##materials and also for chemical ##plants in the industry become . |
| **`30k`**  | Generated        | nun gibt es regeln , die sich aus organis ##atorischen mitteln aus dem organis ##chen , die in der industrie von brennstoff ##nutzung dieser branche stammen , sowie fur die stoffe . |
| -          | Back Translation | Now there are rules that come from the organizational resources that come from the industry's fuel use in this industry, as well as for the materials. |

<br><br>

> **BPE Tokenization Model**

| Vocab_size | Sequence_Type | Sequence |
| :---: | :---: | :--- |
| **`10k`** | Generated        | es gibt vorschriften , die auf die freiwill ige operation en und die , wie es mit brenn stoffen , die fre ise tzung von brenn stoffen , wie sie fur die industrie verwendet werden . |
| -         | Back Translation | There are regulations that apply to voluntary operations and how to deal with fuels, the release of fuels used for industry. |
| **`20k`** | Generated        | jetzt sind vorschriften zur flucht iger mittel aus flucht iger anlage ystem en , die sich als brennstoff be th altung im bereich der brennstoff bekampfung von brennstoff und aus dieser branche nieder gelassen haben . |
| -         | Back Translation | There are now regulations for the escape of volatile substances from volatile plant systems that have established themselves as fuel containment in the field of fuel control and from this industry. |
| **`30k`** | Generated        | es gibt regelungen , die fur flucht nach wie vor die flucht nach einwanderungs verbindungen mit den stoffen , die fur die stoffe in dieser branche verwendet werden . |
| -         | Back Translation | There are regulations in place for escape after immigration connections with the substances that are used for the substances in this industry. |

<br><br>

> **Unigram Tokenization Model**

| Vocab_size | Sequence_Type | Sequence |
| :---: | :---: | :--- |
| **`10k`** | Generated        | e s gib t regelung en , die von der versicherung en , die fur diese art von i o be r n i e s , wie fur die in der industrie vorgesehen en form vorgesehen en regelung en . |
| -         | Back Translation | There are regulations that are provided by the insurance companies for this type of i o r n ie s, as for the form provided for in the industry. |
| **`20k`** | Generated        | derzeit gib t e s regeln , die fur die verbreitung von organische n organische n verbindung en , die sich mit dem brennstoff zu tun , was diese r art der brennstoffe eingesetzt werden . |
| -         | Back Translation | Currently there are rules governing the distribution of organic n organic n compounds that have to do with the fuel, what this type of fuel is used for. |
| **`30k`** | Generated        | e s gib t regeln , die fur die fluchtige n organische r und organische r beihilfen , die sie fur fossile stoffe in betracht ziehen . |
| -         | Back Translation | There are rules governing volatile organic compounds and organic compounds that they consider for fossil fuels. |

<br><br>



## How to Use
```
git clone https://github.com/moon23k/Tokenizers.git
```

```
cd Tokenizers
python3 setup.py -tokenizer_type ['all', 'WL', 'WP', 'BPE', 'UNI'] -vocab_size [10k, 20k, 30k]
python3 run.py -mode [train, test, inference] -tokenizer_type ['WL', 'WP', 'BPE', 'UNI'] -vocab_size [10k, 20k, 30k]
```
</br>

