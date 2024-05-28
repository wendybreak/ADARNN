import torch
import torch.nn as nn
from base.loss_transfer import TransferLoss
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualAdaRNN(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_width=256, n_input=128, n_hiddens=[128, 64], n_output=6, dropout=0.2, len_seq=9, model_type='AdaRNN', trans_loss='mmd', bidirectional=True, rnn_type='GRU'):
        super(DualAdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        self.bidirectional = bidirectional

        rnn_class = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        features = nn.ModuleList()
        for hidden in n_hiddens:
            features.append(rnn_class(input_size=n_input,
                                      num_layers=1,
                                      hidden_size=hidden,
                                      batch_first=True,
                                      dropout=dropout,
                                      bidirectional=bidirectional))
            n_input = hidden * 2 if bidirectional else hidden
        self.features = nn.Sequential(*features)

        final_output_size = n_hiddens[-1] * 2 if bidirectional else n_hiddens[-1]
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(final_output_size, bottleneck_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.ReLU(),
                nn.BatchNorm1d(bottleneck_width)
            )
            self.fc = nn.Linear(bottleneck_width, n_output)
        else:
            self.fc_out = nn.Linear(final_output_size, n_output)

        # Initialize other layers or components you may have added...
    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)
    def get_features(self, output_list):
            fea_list_src, fea_list_tar = [], []
            for fea in output_list:
                fea_list_src.append(fea[0: fea.size(0) // 2])
                fea_list_tar.append(fea[fea.size(0) // 2:])
            return fea_list_src, fea_list_tar

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if self.model_type == 'AdaRNN' else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and not predict:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list
    
    def forward_pre_train(self, x, len_win=0):
        out, out_list_all, out_weight_list = self.gru_features(x)
        fea = out[:, -1, :]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(device)
        for i in range(len(out_list_s)):
            criterion_transfer = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            for j in range(self.len_seq):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (self.len_seq - 1) * (2 * len_win + 1)
                    loss_transfer += weight * criterion_transfer.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list

    def forward_Boosting(self, x, weight_mat=None):
        out, out_list_all, _ = self.gru_features(x)
        fea = out[:, -1, :]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea)
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea).squeeze()

        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(device)
        if weight_mat is None:
            weight_mat = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq)).to(device)
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(device)
        for i in range(len(out_list_s)):
            criterion_transfer = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion_transfer.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer += weight_mat[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight_mat

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]
        x_t = out[int(out.shape[0]//2):]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](
            self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        res = self.softmax(weight).squeeze()
        return res

    def predict(self, x):
        out = self.gru_features(x, predict=True)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
        return fc_out

# Implement other methods like forward_pre_train, forward_Boosting, update_weight_Boosting similarly adjusting for bidirectional outputs
