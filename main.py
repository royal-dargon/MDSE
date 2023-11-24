"""
the start of the work
writen by Jingzhe Li, 2023
"""
import os
import datetime
import warnings

import torch
import torch.cuda
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer
# from fairseq.model.robeata import RobertaModel
from torch.utils.tensorboard import SummaryWriter

import opts
import model
import data
from util.write_file import WriteFile
import train_process


def main():
    warnings.filterwarnings("ignore")
    opt = opts.get_opt()
    # 单卡
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    dt = datetime.datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note
    print('\n', opt.save_model_path, '\n')

    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    loss_func = None
    if opt.loss_type == 'CE':
        loss_func = nn.CrossEntropyLoss()

    work_model = model.Classification(opt)
    if opt.cuda is True:
        assert torch.cuda.is_available()
        work_model = work_model.cuda()
        loss_func = loss_func.cuda()

    print("开始加载数据：")
    tokenizer = None
    abl_path = ''
    if opt.text_model == 'bert-en':
        tokenizer = BertTokenizer.from_pretrained("bert_en/vocab.txt")
    elif opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained("bert_base/vocab.txt")
    elif opt.text_model == 'roberta':
        # tokenizer = RobertaModel.from_pretrained('/bonemodel/roberta.base', checkpoint_file='model.pt')
        vocab_file = './bonemodel/roberta_sa/vocab.json'
        merges_file = './bonemodel/roberta_sa/merges.txt'
        tokenizer = RobertaTokenizer(vocab_file, merges_file)

    if True:
        data_path_root = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_path_name + '/'
        """
        下面分别表示的为训练文本数据的路径
        """
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'dev.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = abl_path + 'dataset/data/' + opt.data_type + '/dataset_image'
        image_coordinate = None
        data_translation_path = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_type + '_translation.json'

    # data_type 表示的为数据类型，1：训练数据；2：开发数据；3：测试数据
    train_loader, opt.train_data_len = data.data_process(opt, train_data_path, tokenizer, photo_path,
                                                        data_type=1, data_translation_path=data_translation_path,
                                                        image_coordinate=image_coordinate)
    dev_loader, opt.dev_data_len = data.data_process(opt, dev_data_path, tokenizer, photo_path, data_type=2,
                                                     data_translation_path=data_translation_path,
                                                     image_coordinate=image_coordinate)
    test_loader, opt.test_data_len = data.data_process(opt, test_data_path, tokenizer, photo_path, data_type=3,
                                                       data_translation_path=data_translation_path,
                                                       image_coordinate=image_coordinate)

    opt.save_model_path = WriteFile(opt.save_model_path, 'train_correct_log.txt', str(opt) + '\n\n', 'a+',
                                    change_file_name=True)
    log_summary_writer = SummaryWriter(log_dir=opt.save_model_path)
    log_summary_writer.add_text('Hyperparameter', str(opt), global_step=1)
    log_summary_writer.flush()

    if opt.run_type == 1:
        # 进入训练模式
        print('\nTraining Begin')
        train_process.train_process(opt, train_loader, dev_loader, test_loader, work_model, loss_func,
                                    log_summary_writer)

    log_summary_writer.close()


if __name__ == "__main__":
    main()
