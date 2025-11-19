import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

user_path=r'../../Ws-Dream/dataset1/userlist.txt'
ws_path=r'../../Ws-Dream/dataset1/wslist.txt'
mat_path='../../Ws-Dream/dataset1/rtMatrix.txt'

def delete_misssing_data(list,mat):
    list_null=np.array(pd.isnull(list.iloc[:,1:]))
    a=np.sum(list_null,1)
    id=np.where(a==0)[0]
    new_list=np.array(list)[id,:]
    new_list[:,0]=np.arange(0,new_list.shape[0])
    new_mat=mat[:,id]
    return new_list,new_mat
def enconde(list):
    feature_col=[]
    a=list[:,:-2]
    b=list[:,-2:]
    c=[]
    for i in range(b.shape[0]):
        c.append(str(int(b[i,0]))+','+str(int(b[i,1])))
    c=np.array(c).reshape((-1,1))
    list=np.concatenate((a,c),1)
    for i in range(list.shape[1]):
        le=LabelEncoder()
        list[:,i]=le.fit_transform(list[:,i])
        feature_col.append(len(le.classes_))
    return list,feature_col

def get_input(user_path,ws_path,mat_path):
    #读取数据
    user_list = np.array(pd.read_csv(user_path, skiprows=2, header=None, sep='\t'))
    ws_list = pd.read_csv(ws_path, skiprows=2, header=None, sep='\t')
    mat = np.array(pd.read_table(mat_path, header=None).iloc[:, :-1])
    #处理空余数据
    ws_list,mat=delete_misssing_data(ws_list,mat)
    #数据进行编码
    user_list,user_cols=enconde(user_list[:,[0,2,4,5,6]])    #country as 经纬度
    ws_list,ws_cols=enconde(ws_list[:,[0,4,6,1,2,7,8]])    #country as ws_provider 经纬度
    return user_list,ws_list,mat,user_cols,ws_cols

def matrix2sequence(data):
    np.set_printoptions(suppress=True)
    n,m=data.shape
    data_list=np.zeros((m*n,3))
    x=np.tile(np.arange(0,n).reshape(-1,1),(1,m)).flatten()
    y=np.tile(np.arange(0,m).reshape(1,-1),(n,1)).flatten()
    data_list[:,0]=x
    data_list[:,1]=y
    data_list[:,2]=data.flatten()
    return data_list

if __name__ == '__main__':
    a,b,c,d,e=get_input(user_path,ws_path,mat_path)
    print(a,b,c)
    a=a.astype(float)
    b=b.astype(float)
