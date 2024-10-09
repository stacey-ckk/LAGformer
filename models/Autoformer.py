import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer, STGCNChebGraphConv,InteractionLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp,TimeSeriesProcessor_bat,TimeSeriesDecomposer
import math
import numpy as np
import pandas as pd
  
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.interaction_layer = InteractionLayer(input_dim=97)  
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.decomposer = TimeSeriesDecomposer(T1=4, T2 = 8)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ], conv_layers=None,#加！
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),#self_attention
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),#cross_attention
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        
        Ko = configs.n_his - (configs.Kt - 1) * 2 * configs.stblock_num

        blocks = []
        blocks.append([1])
        for l in range(configs.stblock_num):
            blocks.append([16, 8, 16])
        if Ko == 0:
            blocks.append([32])
        elif Ko > 0:
            blocks.append([32, 32])
        blocks.append([12])
        
        n_vertex  =97
       
        
        self.sttgcn1 = STGCNChebGraphConv(configs, blocks, n_vertex,flagg = 1)#seasonal
        self.sttgcn2 = STGCNChebGraphConv(configs, blocks, n_vertex,flagg = 2)#trend
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        trend_init, seasonal_init, residual_init  = self.decomposer.decompose(enc_out)
        trend_init_reshaped = trend_init.unsqueeze(1)
        seasonal_init_reshaped = seasonal_init.unsqueeze(1)
        trend_init  = self.sttgcn1(trend_init_reshaped)
        seasonal_init  = self.sttgcn2(seasonal_init_reshaped)
        trend_init = trend_init.squeeze()
        seasonal_init = seasonal_init.squeeze()
        if trend_init.dim() == 2:
            trend_init  = trend_init.unsqueeze(0)
        if seasonal_init.dim() == 2:
            seasonal_init  = seasonal_init.unsqueeze(0)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        residual_init = torch.cat([residual_init[:, -self.label_len:, :], zeros], dim=1)
        dec_out_1 = self.dec_embedding(residual_init, x_mark_dec)
        residual_part, seasonal_part = self.decoder(dec_out_1, enc_out, x_mask=None, cross_mask=None,
                                                 trend=seasonal_init)
        dec_out_2 = self.dec_embedding(seasonal_init_2, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out_2, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        dec_out = trend_part + seasonal_part + residual_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output) 
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :] 
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out 
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out 
        return None
