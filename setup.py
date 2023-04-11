import os, json, yaml, argparse
import sentencepiece as spm



def select_data(orig_data, volumn=32000):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    concat, processed = [], []
    
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
            processed.append(temp_dict)
            concat.append(src + trg)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(concat))

    return processed




def build_vocab(tokenizer_type):
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    assert os.path.exists(f'data/corpus.txt')
    opt = f"--input=data/corpus.txt\
            --model_prefix=data/{tokenizer_type}/tokenizer\
            --vocab_size={vocab_config['vocab_size']}\
            --character_coverage={vocab_config['coverage']}\
            --model_type={tokenizer_type}\
            --pad_id={vocab_config['pad_id']} --pad_piece={vocab_config['pad_piece']}\
            --unk_id={vocab_config['unk_id']} --unk_piece={vocab_config['unk_piece']}\
            --bos_id={vocab_config['bos_id']} --bos_piece={vocab_config['bos_piece']}\
            --eos_id={vocab_config['eos_id']} --eos_piece={vocab_config['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)



def tokenize_data(tokenizer, data_obj):
    max_trg_len = 0
    tokenized = []
    for elem in data_obj:
        temp_dict = dict()
        
        temp_dict['src'] = tokenizer.EncodeAsIds(elem['src'])
        temp_dict['trg'] = tokenizer.EncodeAsIds(elem['trg'])

        if max_trg_len < len(temp_dict['trg']):
            max_trg_len = len(temp_dict['trg'])

        tokenized.append(temp_dict)

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['model'][f'{tokenizer_type}_max_pred_len'] = max_trg_len

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)        
    
    return tokenized


def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)
    


def main(tokenizer_type):
    if os.path.exists('data/corpus.txt')
        orig = load_dataset('wmt14', 'de-en', split='train')['translation']
        selected = select_data(orig)
        save_data(selected)
    
    #Build Vocab
    build_vocab(tokenizer_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tokenizer_type', required=True)
    
    args = parser.parse_args()
    tokenizer_types = ['all', 'word', 'char', 'bpe', 'unigram']
    assert args.tokenizer_type in tokenizer_types
    
    
    if args.tokenizer_type == 'all':
        for tokenizer_type in tokenizer_types[1:]:
            main(tokenizer_type)
    else:
        main(args.tokenizer_type)
