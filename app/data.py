import io

from collections import Counter
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab



def get_filepaths():
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.fr.gz', 'train.en.gz')
    val_urls = ('val.fr.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    return train_filepaths, val_filepaths, test_filepaths



def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])