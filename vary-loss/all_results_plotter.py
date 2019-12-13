import pickle
import matplotlib.pyplot as plt
import pandas as pd
loss_funcs = ['SSIM','L1','L2','SSIML1']
# funcs = ['adamW', 'adam']
train_losses = []
val_losses = []
df = pd.DataFrame({'SSIM': {'train': [], 'val': []}, 'L1': {
                  'train': [], 'val': []}, 'L2': {'train': [], 'val': []}, 'SSIML1': {'train': [], 'val': []}})

for func in loss_funcs:

    df[func]['train'] = pickle.load(
        open(f"./vary-loss/pickles/train_loss_{func}.pkl", 'rb'))
    df[func]['val'] = pickle.load(
        open(f"./vary-loss/pickles/val_loss_{func}.pkl", 'rb'))

# print(df)
plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM')
colours = ['b', 'g', 'r', 'c', 'm', 'y']

for func in loss_funcs:
    for phase in df[func].keys():
        if phase == 'val':
            plt.plot(df[func][phase], label=f"{func} {phase}", linestyle='dashed', color=colours[list(
                df.keys()).index(func)])
        else:
            plt.plot(df[func][phase], label=f"{func} {phase}", color=colours[list(
                df.keys()).index(func)])
# plt.ylim(0,max(TRAIN_LOSS))
plt.title('Loss against epoch number for all loss functions')
plt.legend()
# plt.plot(TRAIN_LOSS,'b-')
# plt.plot(VAL_LOSS,'r-')
# plt.savefig(f"./vary-optim/plots/{func}.png")
plt.show()
