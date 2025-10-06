# coding: utf-8
import sys
import os
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件夹路径
dataset_dir = os.path.join(current_dir, os.pardir, "ptb")  # ptb数据集文件夹路径
sys.path.append(os.path.join(current_dir, os.pardir)) # 添加上级目录到sys.path，方便后续导入其他模块

# 下面的代码假设ptb数据集已经下载好，并放在dataset_dir目录下
url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/' # 仅保留，不再使用 
key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'


def _download(file_name):
    file_path = dataset_dir + '/' + file_name  # 拼接本地文件完整路径
    # 1. 如果文件已存在，直接返回（不做任何操作）
    if os.path.exists(file_path):
        print(f"✅ 本地已找到 {file_name}，跳过下载")
        return
    # 2. 如果文件不存在，报错提示（明确告诉用户缺失哪个文件）
    raise FileNotFoundError(
        f"❌ 未在 {dataset_dir} 中找到 {file_name}！\n"
        f"请确认：1. 你的ptb文件夹路径是否正确（当前路径：{dataset_dir}）\n"
        f"       2. 该文件夹中是否包含 {file_name} 文件"
    )


def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name

    _download(file_name) # 仅检查本地文件是否存在，不下载

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type='train'):
    '''
        :param data_type: 数据的种类：'train' or 'test' or 'valid (val)'
        :return: corpus(词ID序列), word_to_id(词到ID的映射), id_to_word(ID到词的映射)
    '''
    if data_type == 'val': data_type = 'valid'
    save_path = dataset_dir + '/' + save_file[data_type]

    word_to_id, id_to_word = load_vocab()

    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(file_name)  # 仅检查本地文件是否存在，不下载

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    # 运行时会检查本地文件，生成vocab.pkl和npy（若未生成）
    for data_type in ('train', 'val', 'test'):
        print(f"\n正在加载 {data_type} 数据...")
        load_data(data_type)
    print("\n✅ 所有数据加载完成！")
