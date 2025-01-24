import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def roc_plot(target, pred):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(target, pred, pos_label=1)
    roc_score = roc_auc_score(target, pred)
    plt.plot(fpr, tpr, color='tab:red', lw=2)
    plt.plot([0, 1], [0, 1], color='tab:gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('AUC: {:.4f}'.format(roc_score), fontsize=18)
    plt.tight_layout()
    plt.show()

def loss_plot(epochs, losses):
    plt.figure()
    plt.plot(epochs, losses, color = 'orange', lw = 2)
    plt.xlabel('epoch', fontsize = 13)
    plt.ylabel('loss', fontsize = 13)
    plt.title('Loss-curve', fontsize = 18)
    plt.show()

def get_file(person_list, name):
    mod = pd.read_csv('mod.csv')
    data = pd.read_csv('data3.csv', index_col='id')
    for ID in tqdm(person_list):
        forLen = pd.read_csv('20230610\\addM\\' + str(ID) + '.csv')
        mod.loc[len(mod.index)] = [ID, data['touxi'][ID], len(forLen)]
    mod.sort_values(by="len", inplace=True, ascending=False)
    mod.to_csv(str(name) + '.csv')

def creat_batch_data(string, maxBatch):
    # df = pd.read_csv(name + '//' + str(index) + '//' + string)
    df = pd.read_csv('test.csv')
    datas = []
    labels = []
    batch_data = []
    batch_label = []
    last_len = df['len'][0]
    batch_size = 0
    for i in range(len(df)):
        batch_len = df['len'][i]
        if batch_len != last_len or batch_size == maxBatch:
            labels.append(torch.stack(batch_label))
            datas.append(torch.stack(batch_data))
            batch_data = []
            batch_label = []
            batch_size = 0
        df1 = pd.read_csv('20230610//addM//' + str(df['ID'][i]) + '.csv')
        del df1['Unnamed: 0']
        data = torch.from_numpy(df1.values)
        batch_data.append(torch.tensor(data, dtype=torch.float))
        batch_label.append(torch.tensor(df['result'][i], dtype=torch.float))
        last_len = batch_len
        batch_size = batch_size + 1
    if len(batch_data) != 0:
        datas.append(torch.stack(batch_data))
        labels.append(torch.stack(batch_label))
    return datas, labels

def load_data(maxBatch):
    personIDs = pd.read_csv('data3.csv')['id'].tolist()
    seed = 100
    random.seed(seed)
    random.shuffle(personIDs)

    n = len(personIDs)
    train_data_personID = personIDs[: int(n * 0.6)]
    val_data_personID = personIDs[int(n * 0.6) : int(n * 0.8)]
    test_data_personID = personIDs[int(n * 0.8) :]

    get_file(train_data_personID, "train")
    get_file(val_data_personID, "val")
    get_file(test_data_personID, "test")
    get_file(personIDs, "test")

    train_data, train_label = creat_batch_data("train.csv", maxBatch)
    val_data, val_label = creat_batch_data("val.csv", maxBatch)
    test_data, test_label = creat_batch_data("test.csv", maxBatch)

    return train_data, train_label, val_data, val_label, test_data, test_label

class TLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_d = nn.Parameter(torch.Tensor(hidden_size))

        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_g = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def _init_states(self, x):
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t, c_t

    def forward(self, x, init_states=None):
        # batch, sequence, feature
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, 1:-1]
            delta_t = x[:, t, -1]
            cs1_tb = torch.tanh(c_t @ self.W_d + self.b_d)
            time_function = 1 / (delta_t + 0.00001)
            time_function = time_function.unsqueeze(1)
            time_coefficient = time_function.expand(delta_t.size()[0], self.hidden_size)
            time_coefficient = time_coefficient.unsqueeze(0)
            cs2_tb = cs1_tb * time_coefficient
            ct_tb = c_t - cs1_tb
            cx_tb = ct_tb + cs2_tb
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_g + h_t @ self.U_g + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            c_t = f_t * cx_tb + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim = 0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class classifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layersNum, use_cuda):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = layersNum
        self.use_cuda = use_cuda

        self.lstm = TLSTM(input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.act = nn.Sigmoid()

    def forward(self, train_x):
        batchSize = train_x.size()[0]
        lstmOutput, (hn, cn) = self.lstm(train_x)
        hn = hn[0].view(batchSize, self.hidden_dim)
        outputs = self.fc(hn)
        outputs = self.act(outputs)
        # return outputs
        return outputs, hn

def eval(raw_pred, target, ifPlot):
    new_target = []
    for item in target:
        t_list = list(item)
        for i in t_list:
            new_target.append(int(i))
    pred = []
    for item in raw_pred:
        pred.append(item[1] / (item[0] + item[1]))
    auc_num = roc_auc_score(new_target, pred)
    if ifPlot:
        roc_plot(new_target, pred)
    new_pred = np.round(pred)
    acc_num = accuracy_score(new_target, new_pred)
    min_val = min(pred)
    max_val = max(pred)
    if max_val - min_val == 0:
        return [0] * len(pred)
    normalized = [(x - min_val) / (max_val - min_val) for x in pred]
    prob_true, prob_pred = calibration_curve(new_target, normalized, n_bins = 40)
    tmp = pd.DataFrame({'x': prob_pred, 'y': prob_true})
    tmp.to_csv('ecal_data.csv')
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(prob_pred, prob_true, marker='o')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.grid(True)
    plt.show()
    return auc_num, acc_num

def test(model, data, data_label, device, ifPlot):
    model.eval()
    outs = []
    with torch.no_grad():
        for x, y in zip(data, data_label):
            pred_new, hidden_output = model(x)
            pred = torch.sigmoid(pred_new)
            outs.extend(pred.cpu().detach().numpy().tolist())
    result = eval(outs, data_label, ifPlot)

    model.train()
    return result

def train(model, train_data, train_label, val_data, val_label, learning_rate, n_epoch, device):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    loss_function = nn.BCEWithLogitsLoss()

    epochs = []
    losses = []

    best_acc = 0
    best_auc = 0
    auc_list = []
    acc_list = []
    for epoch in tqdm(range(n_epoch)):
        epochs.append(len(epochs))
        all_loss = 0
        for batch_x, batch_y in zip(train_data, train_label):
            output = model(batch_x)
            batch_y = torch.tensor(batch_y, dtype=torch.long)
            batch_y = F.one_hot(batch_y, num_classes = 2)
            batch_y = torch.tensor(batch_y, dtype=torch.float)
            loss = loss_function(output, batch_y)
            all_loss = all_loss + loss
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        all_loss = int(all_loss)
        losses.append(all_loss/len(train_data))
        scheduler.step(epoch)

        auc, acc = test(model, val_data, val_label, device, False)
        auc_list.append(auc)
        acc_list.append(acc)
        if auc > best_auc:
            torch.save(model, 'model.pth')
            best_auc = auc
        if acc > best_acc:
            best_acc = acc
        if epoch % 50 == 0:
            print("")
            print("auc : " + str(auc))
            print("best_auc : " + str(best_auc))
            print("acc : " + str(acc))
            print("best_acc : " + str(best_acc))
    loss_plot(epochs, losses)
    return

if __name__ == '__main__':
    seed = 1209
    torch.manual_seed(seed)

    MAX_BATCH_SIZE = 16
    input_size = 2
    hidden_size = 16
    output_dim = 2
    layers_num = 1
    learning_rate = 5e-3
    epoches = 100
    use_cuda = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, train_label, val_data, val_label, test_data, test_label = creat_batch_data('test.csv', MAX_BATCH_SIZE)
    model = classifier(input_size, hidden_size, output_dim, layers_num, use_cuda)
    train_result = train(model, train_data, train_label, device, False)
    model = torch.load('model.pth')
    test_result = test(model, test_data, test_label, device, True)
    print(test_result)