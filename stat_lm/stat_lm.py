import pickle
import re
from collections import defaultdict
from typing import List, Dict, Optional, Iterable, Tuple
from tqdm import tqdm
import numpy as np
import os

class Tokenizer:
    def __init__(self,
                 eos_token: str = '<EOS>',
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>',
                 vocab_size: int = 1000):

        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_unk_token = '#' + unk_token
        self.end_unk_token = unk_token + '#'
        self.word_unk_token = '#' + unk_token + '#'

        self.special_tokens = [self.pad_token, self.eos_token]
        self.unk_tokens = [self.unk_token, self.start_unk_token, self.end_unk_token, self.word_unk_token]
        self.n_special =len(self.unk_tokens) + len(self.special_tokens)
        self.vocab = dict()
        self.vocab_size = vocab_size
        self.inverse_vocab = dict()
        self.merge_rules = dict()

    def text_preprocess(self, input_text: str) -> str:
        """ Предобрабатываем одно предложение / один текст """
        input_text = input_text.lower()
        input_text = input_text.replace('\n', ' ')
        input_text = re.sub('\s+', ' ', input_text)
        alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' + ' ' + '0123456789'
        out_text = ""
        for elem in input_text:
            if elem in alphabet:
                out_text += elem
        return out_text.strip()

    def _compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split)):
                if i == 0 and split[0][0] != '#':
                    split[0] = '#' + split[0]
                if i == len(split) - 1 and split[-1][-1] != '#':
                    split[-1] = split[-1] + '#'

            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, a, b, splits, word_freqs):

        for word in word_freqs.keys():

            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits


    def build_vocab(self, corpus: List[str]) -> None:
        assert len(corpus)

        word_freqs = defaultdict(int)
        print("Text preprocess...")
        for text in tqdm(corpus):
            text = self.text_preprocess(text)
            for word in text.split():
                word_freqs[word] += 1

        alphabet = []

        for word in word_freqs.keys():
            for i in range(len(word)):
                letter = word[i]
                if letter not in alphabet:
                    alphabet.append(letter)
                    alphabet.append('#' + letter)
                    alphabet.append('#' + letter + '#')
                    alphabet.append(letter + '#')

        vocab = alphabet
        splits = {word: [c for c in word] for word in word_freqs.keys()}

        print('\n Updating vocab...')
        pbar = tqdm(total=self.vocab_size)
        pbar.update(len(alphabet))

        while len(vocab) < self.vocab_size - self.n_special:
            pair_freqs = self._compute_pair_freqs(splits, word_freqs)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if not best_pair:
                break
            splits = self._merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
            pbar.update(1)

        self.vocab = {token: idx for idx, token in enumerate(vocab)}


        for i in range(len(self.unk_tokens)):
            self.vocab[self.unk_tokens[i]] = self.vocab_size - self.n_special + i
            pbar.update(1)

        for i in range(len(self.special_tokens)):
            self.vocab[self.special_tokens[i]] = self.vocab_size - len(self.special_tokens) + i
            pbar.update(1)


        pbar.close()
        self.inverse_vocab = {ind: elem for elem, ind in self.vocab.items()}

    def _tokenize(self, text: str, append_eos_token: bool = True) -> List[int]:
        text = self.text_preprocess(text)
        splits = list()
        for word in text.split():
            split = list()
            for i in range(len(word)):
                letter = word[i]
                if i == 0:
                    letter = '#' + letter
                if i == len(word) - 1:
                    letter = letter + '#'
                split.append(letter)
            splits.append(split)
        for pair, merge in self.merge_rules.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split
        return splits


    def encode(self, text: str, append_eos_token: bool = True) -> List[str]:
        """ Токенизируем текст """
        token_splits = self._tokenize(text, append_eos_token)
        ids = list()
        for split in token_splits:
            for i in range(len(split)):
                unk_token = self.unk_token
                if len(split) == 1 and split[0] not in self.vocab.keys():
                    unk_token = self.word_unk_token
                elif i == 0:
                    unk_token = self.start_unk_token
                elif i == len(split) - 1:
                    unk_token = self.end_unk_token
                ids.append(self.vocab.get(split[i], self.vocab[unk_token]))

        if append_eos_token:
            ids.append(self.vocab[self.eos_token])
        return ids

    def decode(self, input_ids: Iterable[int], remove_special_tokens: bool = True) -> str:
        assert len(input_ids)
        assert max(input_ids) < self.vocab_size and min(input_ids) >= 0
        tokens = []
        j = 0

        for i in range(len(input_ids)):

            token = self.inverse_vocab[input_ids[i]]
            if token in self.special_tokens:
                if not remove_special_tokens:
                    tokens.append(token)
                    j += 1
                continue

            if token[0] == '#' and token[-1] == '#':
                if not remove_special_tokens or token != self.word_unk_token:
                    tokens.append(token[1:-1])
                    j += 1

            elif token[0] == '#':
                if not remove_special_tokens or token != self.start_unk_token:
                    tokens.append(token[1:])

            elif token[-1] == '#':
                if not remove_special_tokens or token != self.end_unk_token:
                    tokens[j] += token[:-1]
                j += 1
            else:
                if not remove_special_tokens or token != self.unk_token:
                    tokens[j] += token

        return ' '.join(tokens)


    def save(self, path: str) -> bool:
        data = {
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'start_unk_token': self.start_unk_token,
            'word_unk_token': self.word_unk_token,
            'end_unk_token': self.end_unk_token,
            'special_tokens': self.special_tokens,
            'unk_tokens': self.unk_tokens,
            'n_special': self.n_special,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'inverse_vocab': self.inverse_vocab,
            'merge_rules': self.merge_rules
        }

        with open(path, 'wb') as fout:
            pickle.dump(data, fout)

        return True

    def load(self, path: str) -> bool:
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        self.eos_token = data['eos_token']
        self.pad_token = data['pad_token']
        self.unk_token = data['unk_token']
        self.start_unk_token = data['start_unk_token']
        self.end_unk_token = data['end_unk_token']
        self.word_unk_token = data['word_unk_token']
        self.special_tokens = data['special_tokens']
        self.unk_tokens = data['unk_tokens']
        self.n_special = data['n_special']
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = data['inverse_vocab']
        self.merge_rules = data['merge_rules']
    

class GenerationConfig:
    def __init__(self, **kwargs):
        """
        Тут можно задать любые параметры и их значения по умолчанию
        Значения для стратегии декодирования decoding_strategy: ['max', 'top-p']
        """
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_tokens = kwargs.pop("max_tokens", 32)
        self.sample_top_p = kwargs.pop("sample_top_p", 0.9)
        self.decoding_strategy = kwargs.pop("decoding_strategy", 'max')
        self.remove_special_tokens = kwargs.pop("remove_special_tokens", False)
        self.validate()

    def validate(self):
        """ Здесь можно валидировать параметры """
        if not (1.0 > self.sample_top_p > 0):
            raise ValueError('sample_top_p')
        if self.decoding_strategy not in ['max', 'top-p']:
            raise ValueError('decoding_strategy')
        
class StatLM:
    def __init__(self,
                 tokenizer: Tokenizer,
                 context_size: int = 2,
                 alpha: float = 0.1
                ):

        assert context_size >= 2

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.alpha = alpha

        self.n_gramms_stat = defaultdict(int)
        self.nx_gramms_stat = defaultdict(int)

    def get_token_by_ind(self, ind: int) -> str:
        return self.tokenizer.vocab.get(ind)

    def get_ind_by_token(self, token: str) -> int:
        return self.tokenizer.inverse_vocab.get(token, self.tokenizer.inverse_vocab[self.unk_token])

    def train(self, train_texts: List[str]):
        for sentence in tqdm(train_texts, desc='train lines'):
            sentence_ind = self.tokenizer.encode(sentence)
            for i in range(len(sentence_ind) - self.context_size):

                seq = tuple(sentence_ind[i: i + self.context_size - 1])
                self.n_gramms_stat[seq] += 1

                seq_x = tuple(sentence_ind[i: i + self.context_size])
                self.nx_gramms_stat[seq_x] += 1

            #здесь добавляется последняя 'n-грамма' (она может быть меньшего размера)
            seq = tuple(sentence_ind[len(sentence_ind) - self.context_size:])
            self.n_gramms_stat[seq] += 1

    def sample_token(self,
                     token_distribution: np.ndarray,
                     generation_config: GenerationConfig) -> int:

        if generation_config.decoding_strategy == 'max':
            return token_distribution.argmax()
        elif generation_config.decoding_strategy == 'top-p':
            token_distribution = sorted(list(zip(token_distribution, np.arange(len(token_distribution)))),
                                        reverse=True)
            total_proba = 0.0
            tokens_to_sample = []
            tokens_probas = []
            for token_proba, ind in token_distribution:
                tokens_to_sample.append(ind)
                tokens_probas.append(token_proba)
                total_proba += token_proba
                if total_proba >= generation_config.sample_top_p:
                    break
            # Чем выше температура T, тем более гладкое (ближе к равномерному) распределение вероятностей
            tokens_probas = np.array(tokens_probas) / generation_config.temperature
            tokens_probas = tokens_probas / tokens_probas.sum()
            return np.random.choice(tokens_to_sample, p=tokens_probas)
        else:
            raise ValueError(f'Unknown decoding strategy: {generation_config.decoding_strategy}')

    def save_stat(self, path: str) -> bool:
        stat = {
            'n_gramms_stat': self.n_gramms_stat,
            'nx_gramms_stat': self.nx_gramms_stat,
            'context_size': self.context_size,
            'alpha': self.alpha
        }
        with open(path, 'wb') as fout:
            pickle.dump(stat, fout)

        return True

    def load_stat(self, path: str) -> bool:
        with open(path, 'rb') as fin:
            stat = pickle.load(fin)

        self.n_gramms_stat = stat['n_gramms_stat']
        self.nx_gramms_stat = stat['nx_gramms_stat']
        self.context_size = stat['context_size']
        self.alpha = stat['alpha']

        return True

    def get_stat(self) -> Dict[str, Dict]:

        n_token_stat, nx_token_stat = {}, {}
        for token_inds, count in self.n_gramms_stat.items():
            n_token_stat[self.tokenizer.decode(token_inds)] = count

        for token_inds, count in self.nx_gramms_stat.items():
            nx_token_stat[self.tokenizer.decode(token_inds)] = count

        return {
            'n gramms stat': self.n_gramms_stat,
            'n+1 gramms stat': self.nx_gramms_stat,
            'n tokens stat': n_token_stat,
            'n+1 tokens stat': nx_token_stat,
        }

    def _get_next_token(self,
                        tokens: List[int],
                        generation_config: GenerationConfig) -> Tuple[int, str]:

        denominator = self.n_gramms_stat.get(tuple(tokens), 0) + self.alpha * len(self.tokenizer.vocab)
        numerators = [0] * self.tokenizer.vocab_size
        for ind in self.tokenizer.inverse_vocab:
            numerators[ind] = self.nx_gramms_stat.get(tuple(tokens + [ind]), 0) + self.alpha

        token_distribution = np.array(numerators) / denominator
        if len(np.unique(token_distribution)) == 1:
            last_token = self.tokenizer.inverse_vocab[tokens[-1]]
            end_status = last_token[-1]
            start_status = last_token[-1][0]
            #После конца слова не может идти ничего, кроме начала следующего (#)
            if end_status == '#' or last_token in self.tokenizer.special_tokens:
                for ind in range(self.tokenizer.vocab_size):
                    token = self.tokenizer.inverse_vocab[ind]
                    if token[0] != '#':
                        token_distribution[ind] = 0

            #После начала слова не может идти начало слова
            elif start_status == '#':
                for ind in range(self.tokenizer.vocab_size):
                    token = self.tokenizer.inverse_vocab[ind]
                    if token[0] == '#':
                        token_distribution[ind] = 0
            #После середины слова не может идти начало слова
            else:
                for ind in range(self.tokenizer.vocab_size):
                    token = self.tokenizer.inverse_vocab[ind]
                    if token[0] != '#':
                        token_distribution[ind] = 0

            #Перенормировка
            token_distribution = token_distribution / token_distribution.sum()

        max_proba_ind = self.sample_token(token_distribution, generation_config)

        next_token = self.tokenizer.inverse_vocab[max_proba_ind]

        return max_proba_ind, next_token

    def generate_token(self,
                       text: str,
                       generation_config: GenerationConfig
                      ) -> Dict:
        tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = tokens[-self.context_size + 1:]

        max_proba_ind, next_token = self._get_next_token(tokens, generation_config)

        return {
            'next_token': next_token,
            'next_token_num': max_proba_ind,
        }


    def generate_text(self, text: str,
                      generation_config: GenerationConfig
                     ) -> Dict:
        
        all_tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = all_tokens[-self.context_size + 1:]

        next_token = None
        while next_token != self.tokenizer.eos_token:

            if  len(all_tokens) == generation_config.max_tokens:
                all_tokens.append(self.tokenizer.vocab[self.tokenizer.eos_token])
                break

            max_proba_ind, next_token = self._get_next_token(tokens, generation_config)
            all_tokens.append(max_proba_ind)
            tokens = all_tokens[-self.context_size + 1:]

        new_text = self.tokenizer.decode(all_tokens, generation_config.remove_special_tokens)

        return new_text

    def generate(self, text: str, generation_config: Dict) -> str:
        return self.generate_text(text, generation_config)

def construct_model():
    config = {
        'temperature': 1e-10,
        'max_tokens': 40,
        'sample_top_p': 0.0011,
        'decoding_strategy': 'top-p',
    }

    BASE_PATH = os.path.dirname(__file__)
    stat_lm_path = os.path.join(BASE_PATH, 'stat_lm.pkl')
    tokenizer_path = os.path.join(BASE_PATH, 'tokenizer.pkl')

    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_path)

    stat_lm = StatLM(tokenizer)
    stat_lm.load_stat(stat_lm_path)

    generation_config = GenerationConfig(temperature=config['temperature'],
                                         max_tokens=config['max_tokens'],
                                         sample_top_p=config['sample_top_p'],
                                         decoding_strategy=config['decoding_strategy'],
                                         remove_special_tokens=True)

    kwargs = {'generation_config': generation_config}
    return stat_lm, kwargs