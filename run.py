import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from layers.Autoformer_EncDec import my_Layernorm, series_decomp,TimeSeriesProcessor,TimeSeriesDecomposer
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from sklearn.model_selection import ParameterGrid
import random
import numpy as np
import pandas as pd
 
if __name__ == '__main__':
    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='Autoformer')
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--data', type=str, required=False, default='exportation', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/data1/zhangjiasheng/little_jobs/DSQ/Time-Series-Library-main/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exportation.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./Time-Series-Library-main/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=6, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--mask_rate', type=float, default=0.2, help='mask ratio')
    parser.add_argument('--anomaly_ratio', type=float, default=0.2, help='prior anomaly ratio (%)')
    parser.add_argument('--top_k', type=int, default=1, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=97, help='encoder input size')#gai
    parser.add_argument('--dec_in', type=int, default=97, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=97, help='output size')
    parser.add_argument('--d_model', type=int, default=97, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=2, help='attn factor')
    parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='sigmoid', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=False)
    parser.add_argument('--channel_independence', type=int, default=1,help='1: channel dependence 0: channel independence for FreTS model')
    parser.add_argument('--num_workers', type=int, default=3, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-04, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAPE', help='loss function')
    parser.add_argument('--lradj', type=str, default='reduce_on_plateau', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--factorlr', type=int, default=0.9, help='the learning rate will be reduced by a factor of 10 (0.1) or half (0.5) when no improvement is seen.')
    parser.add_argument('--threshold', type=int, default= 1e-05, help='measuring an improvement, to focus only on significant changes')
    parser.add_argument('--min_lr', type=int, default= 1e-08, help='This sets the lower bound for the learning rate.')
    parser.add_argument('--patiencelr', type=int, default= 5, help='This sets the lower bound for the learning rate.')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=4, help='gpu')#gai@@
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='9', help='device ids of multile gpus')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64,128,64],help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--time_intvl', type=int, default=3)
    parser.add_argument('--Kt', type=int, default=1)
    parser.add_argument('--heads', type=int, default=2, choices=[4, 3, 2, 1])
    parser.add_argument('--stblock_num', type=int, default=1)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=2, choices=[3, 2, 1])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.15)
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay (L2 penalty)')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    
    #----------------------------------#
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    vel_exp = pd.read_csv('exportation.csv',index_col=0)
    gso_test = vel_exp.corr().values
    gso_test = gso_test.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso_test)
    decomposer = TimeSeriesDecomposer(T1=24, T2 = 12)
    trend_df, seasonal_df, residual_df = decomposer.decompose(vel_exp)
    if trend_df.requires_grad:
        trend_df = trend_df.detach()  
    trend_df_numpy = trend_df.numpy() 
    trend_df = pd.DataFrame(trend_df_numpy)  
    gsoo_1 = trend_df.corr().values
    gsoo_1 = gsoo_1.astype(dtype=np.float32)
    args.gsoo_1 = torch.from_numpy(gso_test)
    if seasonal_df.requires_grad:
        seasonal_df = seasonal_df.detach()  
    seasonal_df_numpy = seasonal_df.numpy() 
    seasonal_df = pd.DataFrame(seasonal_df_numpy)  
    gsoo_2 = seasonal_df.corr().values
    gsoo_2 = gsoo_2.astype(dtype=np.float32)
    args.gsoo_2 = torch.from_numpy(gso_test)
    args.model='Autoformer'
    print('Args in experiment:')
    print_args(args)
if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
else:
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    #torch.cuda.empty_cache()    
