## Tokenization Strategies
자연어 처리는 인간이 사용하는 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능의 한 분야입니다. 
자연어를 컴퓨터가 이해할 수 있는 이진법으로 표현하기 위해서는 우선 자연어 시퀀스를 작은 단위로 분할해야 하며, 이를 위한 일련의 프로세스를 Tokenization이라고 합니다.
Tokenization을 통해 시퀀스를 분할하는 것은 방식에 따라 Word, Character, SubWord Tokenization으로 크게 나누어 볼 수 있습니다.

Word Tokenization은 구분자를 기준으로 시퀀스를 분할하는 것을, Character Tokenization은 특정 언어의 최소 Character단위로 분할하는 것을, 그리고 Sub Word는 Word Tokenization과 Character Tokenization의 중간 형태로 분할 하는 것을 의미합니다.

Word Tokenizer는 구현 방식이 가장 직관적이고, 간단하다는 장점이 있지만, 다양한 도메인의 많은 데이터를 모두 소화하기 위해서는 그만큼 많은 단어 사전이 필요하다는 단점이 존재합니다.
Character Tokenizer는 가장 적은 단어 사전이 필요하지만, Tokenization을 거친 시퀀스의 길이가 매우 길기 때문에 이를 조합하는 모델입장에서는 그만큼 학습이 어렵다는 단점이 존재합니다.
Sub Word Tokenizer는 이 둘의 장단점을 적절하게 섞은 방식으로, 단어를 보다 작은 의미가 있는 단위로 분할해 사용하며, 이를 위한 여러가지 알고리즘이 존재합니다. 구현이 앞선 두 방식보다는 복잡하지만, 가장 대중적으로 사용되는 Tokenzation 방식입니다.

이 repo에서는 Tokenizer에 따라 NMT과제에서 미치는 영향을 파악해봅니다. 이를 위해 네 가지 Tokenization 방식을 사용하며, 각각은 Word-Level, Word-Piece, BPE, Unigram입니다.  


</br></br>


## Tokenizers

**Word-Level Tokenizer** <br>
Word-Level Tokenizer는 텍스트를 공백 문자를 기준으로 단어 단위로 분리하는 가장 기본적인 토큰화 방법입니다. 예를 들어, "This is what you came for"라는 문장은 ["This", "is", "what", "you", "came", "for"]와 같이 단어 단위로 토큰화됩니다. 이 방법은 간단하고 직관적이지만, 언어에 따라 단어의 형태가 다양하게 변할 수 있기 때문에 일관된 처리를 보장하기 어렵습니다. 

<br>


**Word Piece Tokenizer** <br>

<br>

**Byte Pair Encoding Tokenizer** <br>
Byte Pair Encoding(BPE)는 시퀀스에 포함된 단어들을 적절한 단위로 분할하는 Sub-Word Tokenization 알고리즘 중 하나 입니다. BPE 알고리즘은 Token들의 출현 빈도를 기반으로 높은 빈도의 Token들을 Merge하며, 최종적으로 사용자가 기 설정한 Merge Step혹은 Vocab Size에 도달하는 시점까지 분할 및 병합을 반복합니다.
BPE 알고리즘은 간단한 연산 구조를 바탕으로, 빠른 속도로 작동한다는 장점이 있습니다. 하지만 Pre-Tokenization을 가정하고 있기 때문에, 띄어쓰기가 없는 일부 언어에 대해서는 적용이 힘들다는 단점이 존재합니다.
<br>

**Unigram** <br>

</br></br>


## Setups

**Model** <br>

**Data** <br>


</br></br>

## Results
| &emsp; Tokenizer Type &emsp; | &emsp; Vocab Size &emsp; | &emsp; Greedy Score &emsp; | &emsp; Beam Score &emsp; |
|:---:|:---:|:---:|:---:|
| Word Level | 10k |15.31|11.54|
| Word Level | 30k |18.92|12.30|
| Word Piece | 10k |18.78|14.75|
| Word Piece | 30k |16.76|16.71|
| BPE | 10k |20.66|14.08|
| BPE | 30k |15.01|15.37|
| Unigram | 10k |15.34|12.71|
| Unigram | 30k |14.29|14.10|

<br>

Word Level을 제외한 세 가지 Sub-Word Tokenizer는 모두 Vocab Size가 증가함에 따라 성능 하락이 발생했습니다.
이는 비교적 작은 모델과 데이터에서 지나치게 많은 Vocab 선택지를 부여해, 모델의 추론 능력이 하락했다는 것으로 해석할 수 있습니다. 
반면 Word Level에서 Vocab Size 상승에 따른 성능 향상은, 더 많은 단순 단어 표현 가능성 제고로 인한 것으로 해석할 수 있습니다.

</br></br>

## References

<br>
