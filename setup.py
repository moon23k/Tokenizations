import os, json, argparse
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.models import WordLevel, BPE, WordPiece, Unigram
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer, UnigramTrainer




def select_data(orig_data, volumn=101100):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    corpus, selected = [], []
    
    for elem in orig_data:
        temp_dict = dict()
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict['src'] = src
            temp_dict['trg'] = trg

            selected.append(temp_dict)
            corpus.append(src)
            corpus.append(trg)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return selected



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)



def train_tokenizer(tokenizer_type, vocab_size):
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    os.makedirs(f'tokenizer/{tokenizer_type}', exist_ok=True)

    _vocab_size = int(vocab_size[:-1]) * 1000
    special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']


    if tokenizer_type == 'WL':
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=_vocab_size, min_frequency=2, special_tokens=special_tokens)
    
    elif tokenizer_type == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=_vocab_size, min_frequency=2, special_tokens=special_tokens)
    
    elif tokenizer_type == 'WP':
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=_vocab_size, special_tokens=special_tokens)

    elif tokenizer_type == 'UNI':
        tokenizer = Tokenizer(Unigram())
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(vocab_size=_vocab_size, unk_token='[UNK]', special_tokens=special_tokens)


    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"tokenizer/{tokenizer_type}/{tokenizer_type}_{vocab_size}.json")



def main(tokenizer_type, vocab_size, tt_lst, vs_lst):
    #Load, Select and Save Dataset
    orig_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    selected_data = select_data(orig_data)
    save_data(selected_data)

    #Train Tokenizer
    if tokenizer_type == 'all':
        for tt in tt_lst:
            if vocab_size == 'all':
                for vs in vs_lst:
                    train_tokenizer(tt, vs)    
            else:
                train_tokenizer(tt, vocab_size)        
    else:
        train_tokenizer(tokenizer_type, vocab_size)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tokenizer_type', required=True)
    parser.add_argument('-vocab_size', required=True)
    
    tt_lst = ['all', 'WL', 'WP', 'BPE', 'UNI']
    vs_lst = ['all', '5k', '10k', '15k']

    args = parser.parse_args()
    assert args.tokenizer_type in tt_lst
    assert args.vocab_size in vs_lst
    main(args.tokenizer_type, args.vocab_size, tt_lst[1:], vs_lst[1:])
