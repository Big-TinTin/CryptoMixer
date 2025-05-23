import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.GELU()


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    def __init__(self, time_step):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module):
    def __init__(self, time_step, channel, scale_count=1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList([nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2 ** i, stride=2 ** i),
            TriU(int(time_step / 2 ** i)),
            nn.Hardswish(),
            TriU(int(time_step / 2 ** i))
        ) for i in range(scale_count)])
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, scale_dim=8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


# Because the number of transaction users is not fixed, we use 0 to pad it to 512 users, allowing stockmixer to mixing in the user dimension
class NoGraphMixer(nn.Module):
    def __init__(self, hidden_dim=64, pad_length = 512):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(pad_length, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, pad_length)
        self.pad_length = pad_length

    def forward(self, inputs):
        x = inputs
        x = x.permute(0, 2, 1)
        pad_dim = self.pad_length - x.shape[2]
        pad = torch.zeros([x.shape[0], x.shape[1], pad_dim]).to(x)
        x = torch.cat([pad, x], dim=2)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x[:, :, (pad_dim - self.pad_length):]
        x = x.permute(0, 2, 1)
        return x


class StockMixer(nn.Module):
    def __init__(self, input_dim, time_steps = 64):
        super(StockMixer, self).__init__()
        scale_dim = 16
        self.mixer = MultTime2dMixer(time_steps, input_dim, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(input_dim, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, input_dim)
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=4, stride=4)
        self.stock_mixer = NoGraphMixer()
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, input_dim)

        self.y_dim = time_steps * 2 + scale_dim

    def forward(self, x):
        # x [batch, p, seq, dim]
        batch, p, seq, dim = x.size()
        res_x = x.reshape([batch * p, seq, dim])
        x = res_x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(res_x, x)
        y = self.channel_fc(y).squeeze(-1)

        y = y.reshape([batch, p, self.y_dim])
        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return (y + z)[:, -1]