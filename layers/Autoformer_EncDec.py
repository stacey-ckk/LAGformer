import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.nonparametric.smoothers_lowess import lowess

class my_Layernorm(nn.Module):
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)
    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]
    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, moving_avg=6, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        #self.attention = FourierAttention(d_model, num_heads = 1)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,#torch.Size([32, 12, 16])
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x#更多的周期性
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        res = x
        return res, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=6, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, cross, x_mask=None, cross_mask=None):  
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend  = self.dropout(residual_trend)
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class TimeSeriesProcessor:
    def __init__(self, dataframe):
        self.df = dataframe
        self.df = self.df.astype(float)
    def fft_trend_residuals(self, threshold):
        fft_results = np.fft.fft(self.df.values, axis=0)
        n = self.df.shape[0]
        mask = np.zeros(n, dtype=bool)
        mask[:threshold] = 1
        mask[-threshold:] = 1
        low_freq_fft = fft_results.copy()
        low_freq_fft[~mask, :] = 0
        trends = np.fft.ifft(low_freq_fft, axis=0).real
        residuals = self.df.values - trends
        return pd.DataFrame(trends, index=self.df.index, columns=self.df.columns), pd.DataFrame(residuals, index=self.df.index, columns=self.df.columns)


class TimeSeriesProcessor_bat:
    def __init__(self, tensor):
        self.tensor = tensor
    def fft_trend_residuals(self, threshold):
        batch_size, seq_len, num_features = self.tensor.shape
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[:threshold] = 1
        mask[-threshold:] = 1
        mask = mask.expand(batch_size, num_features, seq_len).transpose(1, 2)
        fft_results = torch.fft.fft(self.tensor, dim=1)
        low_freq_fft = fft_results.clone()
        low_freq_fft[~mask] = 0
        trends = torch.fft.ifft(low_freq_fft, dim=1).real
        residuals = self.tensor - trends
        return trends, residuals


class TimeSeriesDecomposer:
    def __init__(self, T1, T2):
        self.T1 = T1
        self.T2 = T2
    def decompose(self, batch_y):
        if isinstance(batch_y, pd.DataFrame):
            batch_y = torch.tensor(batch_y.values)
        trend = torch.zeros_like(batch_y)
        seasonal = torch.zeros_like(batch_y)
        residual = torch.zeros_like(batch_y)
        if batch_y.dim() == 2:
            for j in range(batch_y.shape[1]):
                series = batch_y[:, j].cpu().numpy() 
                x = np.arange(len(series))
                frac_trend = self.T1 / len(series)
                trend_series = lowess(series, x, frac=frac_trend)[:, 1]
                detrended = series - trend_series
                frac_season = self.T2 / len(series)
                seasonal_series = lowess(detrended, x, frac=frac_season)[:, 1]
                residual_series = detrended - seasonal_series
                trend[:, j] = torch.from_numpy(trend_series)
                seasonal[:, j] = torch.from_numpy(seasonal_series)
                residual[:, j] = torch.from_numpy(residual_series)
        else:
            for i in range(batch_y.shape[0]): 
                for j in range(batch_y.shape[2]): 
                    series = batch_y[i, :, j].cpu().detach().numpy() 
                    x = np.arange(len(series))
                    frac_trend = self.T1 / len(series)
                    trend_series = lowess(series, x, frac=frac_trend)[:, 1]
                    detrended = series - trend_series
                    frac_season = self.T2 / len(series)
                    seasonal_series = lowess(detrended, x, frac=frac_season)[:, 1]
                    residual_series = detrended - seasonal_series
                    trend[i, :, j] = torch.from_numpy(trend_series)
                    seasonal[i, :, j] = torch.from_numpy(seasonal_series)
                    residual[i, :, j] = torch.from_numpy(residual_series)
        return trend, seasonal, residual
    