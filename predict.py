import numpy as np
import scipy as sp
import csv
import os
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import KFold
from torch.autograd import backward
import sys

args = sys.argv

############# MODEL ###############
###################################

# model_dir = './model_save/'
model_loss_function = "BRP"
model_n_factor = 128
model_n_negative = 64
model_learning_rate = 0.0001
model_epoch = 42
model_PATH = 'all_loss_{}_fac_{}_neg_{}_lr_{}_e_{}'.format(model_loss_function, model_n_factor, model_n_negative, model_learning_rate, model_epoch)
###################################
################################################################################ PARAMETERS
loss_function = "BRP"
n_epoch = 1
n_factor = model_n_factor
n_negative = 64

learning_rate = 0.0001
weight_decay = 0.0001
momentum = 0.9
# batch_size = 32
# functionlist = ["BRP", "BCE"]
# factorlist = [128, 64, 32, 16]
# factorlist = [32, 128, 64, 16]
# negativelist = [32, 16, 64]
# negativelist = 32
################################################################################ LOAD DATA
f_name = "train.csv"
with open(f_name, mode='r') as f:
    reader = csv.reader(f, delimiter=',')
    reader = reader
    user_id = []
    item_id = []
    for i, r in enumerate(reader):
        if i == 0:
            continue
        user_id.append(int(r[0]))
        item_id.append(list(map(int, r[1].split(' '))))
num_user_id = len(user_id)
num_item_id = max(max(i) for i in item_id) + 1

################################################################################ NEGATIVE SAMPLE
def ng_sample(num_user_id, item_id, n_negative, num_item_id):
    User = []
    item_i = []
    item_j = []
    for u, ids in enumerate(item_id):
        # print("--------------------- negative sampling user : {} ---------------------".format(u))
        for i in ids:
            for t in range(n_negative):
                j = np.random.randint(num_item_id)
                while j in ids:
                    # print("miss random j: {}".format(j))
                    j = np.random.randint(num_item_id)
                User.append(u)
                item_i.append(i)
                item_j.append(j)
                # print(u, i, j)
    return User, item_i, item_j

################################################################################ MODEL
class MF_BPR(nn.Module):
    def __init__(self, num_user_id, num_item_id, n_factor, std=0.01):
        super(MF_BPR, self).__init__()
        self.P = nn.Embedding(num_user_id, n_factor)
        self.Q = nn.Embedding(num_item_id, n_factor)

        nn.init.normal_(self.P.weight, std=std)
        nn.init.normal_(self.Q.weight, std=std)

    def forward(self, u, i, j):
        P_u = self.P(u)
        Q_i = self.Q(i)
        Q_j = self.Q(j)
        # print("P_u.shape: {}".format(P_u.shape))
        # print("Q_i.shape: {}".format(Q_i.shape))
        pred_i = (P_u * Q_i).sum(dim=1)
        pred_j = (P_u * Q_j).sum(dim=1)
        # print("pred_i: {}".format(pred_i))
        # print("type(pred_i): {}".format(type(pred_i)))
        # print("pred_i.shape: {}".format(pred_i.shape))
        return pred_i, pred_j

    def get_matrics(self, user_id, num_item_id):
        # matrics = torch.mm(torch.Tensor(self.P),torch.Tensor(self.Q))
        P = self.P(torch.LongTensor(
            [i for i in range(len(user_id))]))
        # print(P.shape)
        # a = [i for i in range(len(num_item_id))]
        # print(len(a))
        Q = self.Q(torch.LongTensor(
            [i for i in range(num_item_id)]))
        # print(Q.shape)
        matrics = torch.mm(P, Q.T)

        # matrics = torch.sum(self.P * self.Q, 1)
        # print("P.shape".format(P.shape))
        # print("Q.shape".format(Q.shape))
        # print("matrics.shape: {}".format(matrics.shape))
        return matrics

################################################################################ TRAIN


model = MF_BPR(num_user_id, num_item_id, n_factor)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)
model.load_state_dict(torch.load(model_PATH))


# matrix = csr_matrix(matrix)
# print(matrix)

log_list = []
for epoch in range(n_epoch):
    print("--------------------- epoch : {} ---------------------".format(epoch))
    # # print("negative sampling......")
    # u, i, j = ng_sample(num_user_id, item_id, n_negative, num_item_id)
    # # print(triples)
    # # triples = torch.LongTensor(triples)
    # # train_loader = torch.utils.data.DataLoader(train, batch_size=len(triples), shuffle=False)
    # u = torch.LongTensor(u)
    # i = torch.LongTensor(i)
    # j = torch.LongTensor(j)
    # # train = torch.utils.data.TensorDataset(u,i,j)
    # # train_loader = torch.utils.data.DataLoader(train, batch_size=len(u), shuffle=False)
    # model.train()
    # train_loss = 0
    # # print("training...")
    # # for u,i,j in train_loader:
    # # print("u: {}, i: {}, j: {}".format(u,i,j))
    # optimizer.zero_grad()
    # # print("len(i): {}".format(len(i)))
    # pred_i, pred_j = model(u, i, j)
    # if loss_function == "BCE":
    #     pred_i = torch.sigmoid(pred_i)
    #     pred_j = torch.sigmoid(pred_j)
    #     loss = nn.BCELoss()(pred_i, torch.ones(len(pred_i)))
    #     loss += nn.BCELoss()(pred_j, torch.zeros(len(pred_j)))
    # else:
    #     # loss = -(torch.Tensor(pred_i) - torch.Tensor(pred_j)).sigmoid().log().sum() / u.shape[0]
    #     loss = -(torch.Tensor(pred_i) -
    #                 torch.Tensor(pred_j)).sigmoid().log().sum()
    # # print("backwarding...")
    # loss.backward()
    # # print("optimize stepping...")
    # optimizer.step()
    # # train_loss += loss
    # # print("train_loss: {}".format(train_loss))
    # # print("type(train_loss: {}".format(type(train_loss.detach().numpy())))

################################################################################ GET RANKING
    # print("get ranking...")
    # print(num_item_id)
    ranking_list = [["UserId", "ItemId"]]
    matrics = model.get_matrics(
        user_id, num_item_id).detach().numpy()
    for u_id, ids in enumerate(item_id):
        ranking = matrics[u_id]
        ranking[ids] = -1
        ranking = ranking.argsort()[::-1]
        ranking = list(ranking[:50])
        # print(ranking)
        # print(type(ranking))
        # print(type(u_id))
        # print(type(" ".join(map(str, ranking))))
        # print(ranking.shape)
        ranking_list.append(
            [int(u_id), " ".join(map(str, ranking))])
################################################################################ CALCURATE MAP
    # MAP = 0
    # for index, ids in enumerate(ranking_list[1]):
    #     if ids == "ItemId":
    #         continue
    #     else:
    #         user = index - 1  # because index == 1 is "UserId" in ranking_list

    #         ap = 0
    #         count = 1
    #         for r, id_ in enumerate(ids):
    #             rank = r + 1
    #             if id_ in val_set[user]:
    #                 ap += count / rank
    #                 count += 1
    #             ap = ap / len(val_set[user])
    #         MAP += ap
    # MAP /= len(num_user_id)
    # print("MAP: {}".format(MAP))
    # log_list.append([epoch, train_loss.detach().numpy(), MAP])

################################################################################ SAVE OUTPUT
    # output_dir = './output_save/'
    output_PATH = args[1]
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    with open(output_PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(ranking_list)
    print("Saved output!!")

################################################################################ SAVE MODEL
    # model_dir = './model_save/'
    # model_PATH = '{}loss_{}_fac_{}_neg_{}_lr_{}_e_{}'.format(model_dir, loss_function, n_factor , n_negative, learning_rate, epoch)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # torch.save(model.state_dict(), model_PATH)
    # print("Saved model!!")
    # if loss == "nan":
    #     break

################################################################################ SAVE LOG_LOSS
# log_dir = './log_save/'
# log_PATH = '{}log_loss_{}_fac_{}_neg_{}_lr_{}.csv'.format(log_dir, loss_function, n_factor , n_negative,learning_rate )
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# with open(log_PATH, 'w') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerows(log_list)
#     print("Saved log file!!")
