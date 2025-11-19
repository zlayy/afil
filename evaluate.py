import numpy as np
import pandas as pd
import torch
import csv
def metrics(model, dataloader,loss_func):
    with torch.no_grad():
        error1=0.
        error2=0.
        test_loss=0.
        for user,ws,target in dataloader:
            target=target.reshape((-1,1))
            if torch.cuda.is_available():
                user,ws,lable=user.cuda(),ws.cuda(),target.cuda()
            pred = model(user,ws)
            loss=loss_func(pred,target)
            test_loss+=loss.item()
            error1+=np.abs(pred-target).sum()
            error2+=(pred-target).pow(2).sum()
        mae=error1/len(dataloader.dataset)
        rmse=np.sqrt(error2/len(dataloader.dataset))
        return mae,rmse

def save_result(file_name,args,df_history):
    file=open(f'{file_name}.txt','a')
    df_history = np.array(df_history)
    file.write(f'{args}\n')
    file.write('============ Detailed results ============\n')
    file.write('| epoch | train_loss | test_loss | mae | rmse |\n')
    np.savetxt(file,df_history,fmt='%.4f',delimiter='    ')
    file.close()
    mae,rmse=df_history[-1,-2],df_history[-1,-1]
    #文件写入csv
    with open('result.csv','a',encoding='utf-8',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow([f'%{args.md*100:.1f}',f'{args.n_head}',f'{args.embed_dim}',f'{args.d_k}'             ,f'{args.d_ff}',f'{args.learning_rate}',f'{mae:.4f}',f'{rmse:.4f}',f'{args.hidden_units}',f'{file_name}'])

# if __name__ == '__main__':
#     with open('result.csv','w',newline='') as csvfile:
#         writer=csv.writer(csvfile)
#         writer.writerow(['density','n_head','embed_dim','d_k','d_ff','lr','mae','rmse','hidden_units','file'])
    
    


