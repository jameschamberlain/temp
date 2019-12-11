import pickle
import matplotlib.pyplot as plt
import pandas as pd

lrs = ['0.01','0.001']#'0.1','0.0001','0.00001']

train_losses = []
val_losses = []
df=pd.DataFrame({'0.01': {'train':[], 'val':[]}, '0.001': {'train':[], 'val':[]} }) 
# '0.1': {'train':[], 'val':[]}, '0.0001': {'train':[], 'val':[]}, '0.00001': {'train':[], 'val':[]} })

for lr in lrs:
    df[lr]['train'] = pickle.load(open(f"./vary-lr/pickle/UNET-B14-e30-lr{lr}-ssim-adam-train.pkl",'rb'))
    df[lr]['val'] = pickle.load(open(f"./vary-lr/pickle/UNET-B14-e30-lr{lr}-ssim-adam-val.pkl", 'rb'))

#print(df)
plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM')
colours = ['b','g','r','c','m']

for lr in lrs:
    for phase in df[lr].keys():
        if phase == 'val':
            plt.plot(df[lr][phase], label=f"{lr} {phase}",linestyle='dashed',color=colours[list(df.keys()).index(lr)])
        else:
            plt.plot(df[lr][phase], label=f"{lr} {phase}",color=colours[list(df.keys()).index(lr)])
# plt.ylim(0,max(TRAIN_LOSS))
plt.title('Loss against epoch number for all learning rate values')
plt.legend()
# plt.plot(TRAIN_LOSS,'b-')
# plt.plot(VAL_LOSS,'r-')
# plt.savefig(f"./vary-optim/plots/{optimiser}.png")
plt.savefig(f"./vary-lr/plots/best.png")
plt.show()