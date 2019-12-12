import pickle
import matplotlib.pyplot as plt
import pandas as pd
optimisers = ['adagrad', 'adam', 'adamax', 'adamW', 'ASGD', 'SGD']
# optimisers = ['adamW', 'adam']
train_losses = []
val_losses = []
df = pd.DataFrame({'adagrad': {'train': [], 'val': []}, 'adam': {'train': [], 'val': []}, 'adamax': {'train': [], 'val': [
]}, 'adamW': {'train': [], 'val': []}, 'ASGD': {'train': [], 'val': []}, 'SGD': {'train': [], 'val': []}})
# df = pd.DataFrame({'adamW': {'train': [], 'val': []},
#                    'adam': {'train': [], 'val': []}})
for optimiser in optimisers:

    df[optimiser]['train'] = pickle.load(
        open(f"./vary-optim/pickles/train_loss_{optimiser}.pkl", 'rb'))
    df[optimiser]['val'] = pickle.load(
        open(f"./vary-optim/pickles/val_loss_{optimiser}.pkl", 'rb'))

# print(df)
plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM')
colours = ['b', 'g', 'r', 'c', 'm', 'y']

for optimiser in optimisers:
    for phase in df[optimiser].keys():
        if phase == 'val':
            plt.plot(df[optimiser][phase], label=f"{optimiser} {phase}", linestyle='dashed', color=colours[list(
                df.keys()).index(optimiser)])
        else:
            plt.plot(df[optimiser][phase], label=f"{optimiser} {phase}", color=colours[list(
                df.keys()).index(optimiser)])
# plt.ylim(0,max(TRAIN_LOSS))
plt.title('Loss against epoch number for all optimisers')
plt.legend()
# plt.plot(TRAIN_LOSS,'b-')
# plt.plot(VAL_LOSS,'r-')
# plt.savefig(f"./vary-optim/plots/{optimiser}.png")
plt.show()
