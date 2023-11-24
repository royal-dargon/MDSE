"""Read program running parameters"""

import argparse


def get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('-run_type', type=int,
                       default=1, help='1: train, 2: debug train, 3: dev, 4: test')
    parse.add_argument('-save_model_path', type=str,
                       default='checkpoint', help='save the good model.pth path')
    parse.add_argument('-add_note', type=str, default='MVSA-single', help='Additional instructions when saving files')
    parse.add_argument('-data_type', type=str, default='MVSA-single',
                       help='Train data type: MVSA-single and MVSA-multiple and twitter-2015 and twitter-2017')
    parse.add_argument('-gpu_num', type=str, default='1', help='gpu index')
    parse.add_argument('-epoch', type=int, default=50, help='train epoch num')
    parse.add_argument('-batch_size', type=int, default=32,
                       help='batch size number')
    parse.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    parse.add_argument('-min_lr', type=float,
                       default=1e-9, help='the minimum lr')
    parse.add_argument('-acc_grad', type=int, default=1, help='Number of steps to accumulate gradient on '
                                                              '(divide the batch_size and accumulate)')
    parse.add_argument('-num_workers', type=int, default=8,
                       help='loader dataset thread number')
    parse.add_argument('-l_dropout', type=float, default=0.2,
                       help='classify linear dropout')
    parse.add_argument('-train_log_file_name', type=str,
                       default='train_correct_log.txt', help='save some train log')
    parse.add_argument('-optim_b1', type=float, default=0.9,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-optim_b2', type=float, default=0.98,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-data_path_name', type=str, default='10-flod-1',
                       help='train, dev and test data path name')
    parse.add_argument('-word_length', type=int,
                       default=200, help='the sentence\'s word length')
    parse.add_argument('-save_acc', type=float, default=-1, help='The default ACC threshold')
    parse.add_argument('-save_F1', type=float, default=-1, help='The default F1 threshold')
    parse.add_argument('-text_model', type=str, default='roberta', help='language model:bert-en, clip')
    parse.add_argument('-loss_type', type=str, default='CE', help='Type of loss function')
    parse.add_argument('-optim', type=str, default='adamw', help='Optimizer:adam, sgd, adamw')
    parse.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')
    parse.add_argument('-image_model', type=str, default='vit',
                       help='Image model: resnet-18, resnet-34, resnet-50, resnet-101, resnet-152, vit')
    parse.add_argument('-image_size', type=int, default=224, help='Image dim')
    parse.add_argument('-image_output_type', type=str, default='all',
                       help='"all" represents the overall features and regional features of the picture, '
                            'and "CLS" represents the overall features of the picture')
    parse.add_argument('-text_length_dynamic', type=int, default=0, help='1: Dynamic length; 0: fixed length')
    parse.add_argument('-fuse_type', type=str, default='att', help='att, ave, max')
    parse.add_argument('-tran_dim', type=int, default=768,
                       help='Input dimension of text and picture encoded transformer')
    parse.add_argument('-tran_num_layers', type=int, default=4, help='The layer of transformer')
    parse.add_argument('-image_num_layers', type=int, default=2, help='The layer of image transformer')
    parse.add_argument('-cl_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-cl_self_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-temperature', type=float, default=0.07,
                       help='Temperature used to calculate contrastive learning loss')
    parse.add_argument('-train_fuse_model_epoch', type=int, default=20,
                       help='The number of epoch of the model that only trains the fusion layer')

    parse.add_argument('-cuda', action='store_true', default=True,
                       help='if True: use cuda. if False: use cpu')
    parse.add_argument('-fixed_image_model', action='store_true', default=True, help='是否固定图像模型的参数')

    opt = parse.parse_args()
    print(opt)
    return opt

