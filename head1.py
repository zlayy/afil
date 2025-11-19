#去掉多头注意力层中的fc
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import argparse
from preprocessing import *
import os
from load_data import get_list
from evaluate import save_result
import csv
import copy
#文件名
file_name=os.path.basename(__file__).split(".")[0]

mat_path = r'../../Ws-Dream/dataset1/rtMatrix.txt'
user_path = r'../../Ws-Dream/dataset1/userlist.txt'
ws_path = r'../../Ws-Dream/dataset1/wslist.txt'

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device:{device}')

torch.set_default_tensor_type(torch.DoubleTensor)


parser=argparse.ArgumentParser()
parser.add_argument('--md',type=float,default=0.1)
parser.add_argument('--embed_dim','-e',type=int,default=128)
parser.add_argument('--n_head',type=int,default=6)
parser.add_argument('--d_k',type=int,default=128)
parser.add_argument('--hidden_units',type=list,default=[1024,1024,1024])
parser.add_argument('--batch_size','-b',type=int,default=128)
parser.add_argument('--learning_rate','-l',type=float,default=0.001)
parser.add_argument('--cuda','-c',type=int,default=3)
parser.add_argument('--operation','-op',type=str,default='result')
parser.add_argument('--loss_mode','-loss',type=int,default=1)
parser.add_argument('--seed','-s',type=int,default=0)
parser.add_argument('--dropout','-d',type=float,default=0.)
parser.add_argument('--weight_decay',type=float,default=1e-6)
parser.add_argument('--type',type=str,default='rt')
parser.add_argument('--d_ff',type=int,default=48)
args=parser.parse_args()

ws_path = r'../../Ws-Dream/dataset1/wslist.txt'
user_path = r'../../Ws-Dream/dataset1/userlist.txt'
if args.type == 'rt':
    mat_path = r'../../Ws-Dream/dataset1/rtMatrix.txt'
else:
    mat_path = r'../../Ws-Dream/dataset1/tpMatrix.txt'

user_list,ws_list,mat,user_cols,ws_cols=get_input(user_path,ws_path,mat_path)  #mat(339,4612)
user_list=torch.from_numpy(user_list.astype(int))
ws_list=torch.from_numpy(ws_list.astype(int))
fea_num=user_cols+ws_cols
# mat(339*4612)
# 划分训练集和测试集
data_list = matrix2sequence(mat)
# print(data_list)

class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=1):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
    #在训练过程中每次调用需要
    def forward(self,model,x1,x2):
        self.weight_list = self.get_weight(model,x1,x2)
        reg_loss = self.regularization_loss(self.weight_list,self.weight_decay,p=self.p)
        return reg_loss
    #确定设备
    def to(self, device):
        self.device = device
        super().to(device)
        return self
    #得到正则化参数
    def get_weight(self, model,x1,x2):
        weight_list = []
        weight1_list=[]
        weight2_list=[]
        user=user_list[x1].to(device)
        ws=ws_list[x2].to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                if 'embed_layer1' in name:
                    weight1_list.append(weight)
                elif 'embed_layer2' in name:
                    weight2_list.append(weight)
                else:
                    weight_list.append(weight)
        for i,(name,w) in enumerate(weight1_list):
            weight=w[user[:,i].to(torch.long)]
            weight_list.append((name,weight))
        for j,(name,w) in enumerate(weight2_list):
            weight=w[ws[:,j].to(torch.long)]
            weight_list.append((name,weight))
        return weight_list
    #求取正则化的和
    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss

class Dnn(nn.Module):
    def __init__(self,hidden_units,dropout=0.):
        super(Dnn,self).__init__()
        self.dnn_network=nn.ModuleList(nn.Linear(layer[0],layer[1]) for layer in list(zip(hidden_units[:-1],hidden_units[1:])))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        for layer in self.dnn_network:
            x=layer(x)
            x=F.relu(x)
        x=self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,n_head):
        super (MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.d_k=d_k
        self.n_head=n_head
        self.w_k=nn.Linear(d_model,d_k*n_head,bias=False)
        self.w_q=nn.Linear(d_model,d_k*n_head,bias=False)
        self.w_v=nn.Linear(d_model,d_k*n_head,bias=False)
        nn.init.normal_(self.w_k.weight,0,0.1)
        nn.init.normal_(self.w_q.weight,0,0.1)
        nn.init.normal_(self.w_v.weight,0,0.1)
#         self.fc=nn.Linear(d_k,d_model)
        self.layernorm=nn.LayerNorm(d_model)
    def forward(self,x):   #batch,s,d_model
        residual=x  #复制一份输入与经过selfattention层的context拼接
        batch_size=x.shape[0]
        #(batch,s,d_k*n_head)-->(batch,n_head,s,d_k)
        K = self.w_k(x).view((batch_size,-1,self.n_head,self.d_k)).transpose(1,2)
        Q = self.w_k(x).view((batch_size, -1, self.n_head, self.d_k)).transpose(1, 2)
        V = self.w_k(x).view((batch_size, -1, self.n_head, self.d_k)).transpose(1, 2)
        #接下来进行scaledotproduction
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(self.d_k) #(batch,n_head,s,s)
        #进行softmax
        atts=nn.Softmax(dim=-1)(scores)  #(batch,n_head,s,s)
        #atts和v相乘
        context=torch.matmul(atts,V)  #(batch,n_head,s,d_k)
        #转换格式(batch,n_head,s,d_k)-->(batch,n_head,s*d_k)
        context=context.squeeze()
#         cross_interation = 1 / 2 * (torch.pow(torch.sum(context, 1), 2) - torch.sum(torch.pow(context, 2), 1)) #(batch,s*d_k)
#         cross_interation=cross_interation.reshape((batch_size,-1,self.d_k))
#         #进行全连接
#         context=self.fc(cross_interation)  #batch,s,d_model
        return self.layernorm.to(device)(context)


class Net(nn.Module):
    def __init__(self,user_list,ws_list,d_k,n_head,fea_num,embed_dim,hidden_units,dropout=0.):
        '''
        :param user_list: 用户context列表 Ucountry Uas Uregion
        :param ws_list: ws context列表
        :param fea_num: 特征的总数，为一个列表
        :param embed_dim:
        :param hidden_units: dnn参数
        :param dropout: dnn参数
        '''
        super(Net,self).__init__()
        self.user_list=user_list
        self.ws_list=ws_list
        self.p=nn.Parameter(torch.rand((1,)))
        self.embed_layer1=nn.ModuleDict({'embed_user'+str(i):nn.Embedding(feat,embed_dim) for i,feat in enumerate(fea_num[:4])})
        self.embed_layer2=nn.ModuleDict({'embed_ws'+str(i):nn.Embedding(feat,embed_dim) for i,feat in enumerate(fea_num[4:])})
        fm_input=len(fea_num)*embed_dim
        hidden_units.insert(0,fm_input)
        self.dnn=Dnn(hidden_units,dropout)
        self.final_layer=nn.Linear(hidden_units[-1],1)
        self.self_attention=MultiHeadAttention(embed_dim,d_k,n_head)
    def forward(self,x1,x2):  #batch,3
#         mf_out=(self.mf(x1,x2)).reshape((-1,1))
        batch_size=x1.shape[0]
        user=self.user_list[x1].to(device)
        ws=self.ws_list[x2].to(device)
        embed_user=torch.stack([self.embed_layer1['embed_user'+str(i)](user[:,i]) for i in range(user.shape[1])]) #3,batch,embed_dim
        embed_ws=torch.stack([self.embed_layer2['embed_ws'+str(i)](ws[:,i]) for i in range(ws.shape[1])])
        embed_user=torch.transpose(embed_user,1,0) #batch,3,embed_dim
        embed_ws=torch.transpose(embed_ws,1,0)
        input=torch.cat((embed_user,embed_ws),1)  #batch,6,embed_dim
        att=self.self_attention(input)
        encode=att.reshape((batch_size,-1))
        deep=self.final_layer(self.dnn(encode))
        return deep

def run(args):
    print(args)
    args1=args
    train_list, test_list = get_list(mat, args.md, args.seed)
    print('train set lenth is :{}{}test set lenth is :{}'.format(len(train_list), '\n', len(test_list)))

    # 注意索引的类型应该为torch.long
    train_dataset = TensorDataset(torch.tensor(train_list[:, 0]).to(torch.long),
                                  torch.tensor(train_list[:, 1]).to(torch.long), torch.tensor(train_list[:, 2]))
    test_dataset = TensorDataset(torch.tensor(test_list[:, 0]).to(torch.long),
                                 torch.tensor(test_list[:, 1]).to(torch.long), torch.tensor(test_list[:, 2]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True,num_workers=8)
    model = Net(user_list, ws_list, args.d_k,args.n_head, fea_num, args.embed_dim, copy.deepcopy(args.hidden_units),
                args.dropout)
    model = model.to(device)
    reg_loss = Regularization(model, args.weight_decay, p=1).to(device)
    # print(model)
    # loss_func=nn.MSELoss(reduction='sum').to(device)
    if args.loss_mode == 1:
        loss_func = nn.L1Loss(reduction='sum').to(device)
    elif args.loss_mode == 2:
        loss_func = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
    num_epochs = 50
    df_history = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss', 'mae', 'rmse'])
    for epoch in range(1, num_epochs + 1):
        # 开始训练
        model.train()
        loss_sum = 0.
        for step, (x1, x2, target) in enumerate(train_loader, 1):
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            batch_size = x1.shape[0]
            target = target.reshape((-1, 1))
            pred = model(x1, x2)
            loss = loss_func(pred, target) + reg_loss(model, x1, x2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            loss_sum += loss.item()
            if step % 400 == 0:
                print(
                    f'epoch:[{epoch}]/[{num_epochs}] step:[{step}]/[{len(train_loader)}] loss:{loss_sum / (step * batch_size):>.6f}')
        # 更新lr
        scheduler.step()
        # 开始测试
        if epoch % 50==0:
            model.eval()
            test_loss = 0.
            error0 = 0
            error1 = 0
            with torch.no_grad():
                for test_step, (test_x1, test_x2, test_target) in enumerate(test_loader, 1):
                    test_x1, test_x2, test_target = test_x1.to(device), test_x2.to(device), test_target.to(device)
                    test_target = test_target.reshape((-1, 1))
                    test_pred = model(test_x1, test_x2)
                    test_loss += loss_func(test_pred, test_target).item()
                    error0 += (torch.abs(test_pred - test_target)).sum().item()
                    error1 += (test_pred - test_target).pow(2).sum().item()
                mae = error0 / len(test_loader.dataset)
                rmse = np.sqrt(error1 / len(test_loader.dataset))
                info = (epoch, loss_sum / len(train_loader.dataset), test_loss / len(test_loader.dataset), mae, rmse)
                print('==================' * 6)
                print(
                    f'epoch:[{epoch}]/[{num_epochs}],lr:{scheduler.get_last_lr()},test_loss:{test_loss / len(test_loader.dataset):>.6f},mae:{mae:>.4f},rmse:{rmse:>.4f}')
                df_history.loc[epoch / 50] = info
                print('==================' * 6)
    df_history.reset_index(drop=True)
    print(df_history)
    # 保存结果到txt
    # 文件名
    file_name = os.path.basename(__file__).split(".")[0]
    file_name = f'{file_name}_{args.operation}_{args.md}'
    save_result(file_name, args1, df_history)
#去掉了mha中的fc层
for args.md in [0.025,0.05,0.075,0.1]:
    args.n_head=1
    run(args)


