import json
import copy

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm
import torch
import torch.nn.utils.rnn as run_utils
import numpy as np

import opts


class TwitterDateset(Dataset):
    def __init__(self, opt, data_path, text_tokenizer, photo_path, image_transforms, data_type):
        self.data_type = data_type
        self.dataset_type = opt.data_type
        self.photo_path = photo_path
        self.image_transforms = image_transforms

        file_read = open(data_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()

        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        for data in file_content:
            self.data_id_list.append(data['id'])
            self.text_list.append(data['text'])
            self.label_list.append(data['emotion_label'])

        self.image_id_list = [str(data_id) for data_id in self.data_id_list]

        self.text_token_list = [text_tokenizer.tokenize('<s>' + text + '</s>') for text in
                                tqdm(self.text_list, desc='convert text to token')]
        self.text_token_list = [text if len(text) < opt.word_length else text[0:opt.word_length]
                                for text in self.text_token_list]
        self.text_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in
                           tqdm(self.text_token_list, desc='convert text to id')]

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        image_path = self.photo_path + '/' + str(self.data_id_list[index])
        image_read = Image.open(image_path)
        image_read = image_read.convert("RGB")
        image_read.load()
        image_origin = self.image_transforms(image_read)
        return self.text_to_id[index], image_origin, self.label_list[index]


class Collate:
    def __init__(self, opt):
        self.text_length_dynamic = opt.text_length_dynamic
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.word_length

        self.image_mask_num = 0
        if opt.image_output_type == 'cls':
            self.image_mask_num = 1
        elif opt.image_output_type == 'all':
            self.image_mask_num = 49

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
        image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
        label = torch.LongTensor([b[2] for b in batch_data])

        data_length = [text.size(0) for text in text_to_id]

        max_length = max(data_length)
        if max_length < self.min_length:
            # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
            text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
            max_length = self.min_length

        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)

        bert_attention_mask = []
        text_image_mask = []
        for length in data_length:
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_length - length))
            bert_attention_mask.append(text_mask_cell[:])

            text_mask_cell.extend([1] * self.image_mask_num)
            text_image_mask.append(text_mask_cell[:])

        temp_labels = [label - 0, label - 1, label - 2]
        target_labels = []
        for i in range(3):
            temp_target_labels = []
            for j in range(temp_labels[0].size(0)):
                if temp_labels[i][j] == 0:
                    temp_target_labels.append(j)
            target_labels.append(torch.LongTensor(temp_target_labels[:]))

        return text_to_id, torch.LongTensor(bert_attention_mask), image_origin, torch.LongTensor(text_image_mask), label, target_labels


def get_resize(image_size):
    for i in range(20):
        if 2**i >= image_size:
            return 2**i
    return image_size


def data_process(opt, data_path, text_tokenizer, photo_path, data_type):
    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = TwitterDateset(opt, data_path, text_tokenizer, photo_path,
                             transform_base if data_type == 1 else transform_test_dev, data_type)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt),
                             pin_memory=True if opt.cuda else False)

    return data_loader, dataset.__len__()


def test():
    o = opts.get_opt()
    o.acc_batch_size = 64
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased/bert-base-uncased-vocab.txt")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_loader, o.test_data_len = data_process(o, data_path='dataset/data/twitter-2015/dev.json',
                                                text_tokenizer=tokenizer,
                                                photo_path='dataset/data/twitter-2015/twitter_images', data_type=1)
    test_loader_tqdm = tqdm(test_loader, desc='Train Iteration:')
    for index, data in enumerate(test_loader_tqdm):
        text_origin, bert_attention_mask, image_origin, text_image_mask, label = data
        print(label.shape)

if __name__ == "__main__":
    test()







