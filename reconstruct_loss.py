import pickle
import matplotlib.pyplot as plt

optimiser = "adam"

TRAIN_LOSS = pickle.load(open(f"./vary-optim/pickles/train_loss_{optimiser}.pkl",
                              'rb'))
VAL_LOSS = pickle.load(open(f"./vary-optim/pickles/val_loss_{optimiser}.pkl", 'rb'))

plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM')
# plt.ylim(0,max(TRAIN_LOSS))
plt.title(f'Using {optimiser} optimiser')
plt.plot(TRAIN_LOSS,'b-')
plt.plot(VAL_LOSS,'r-')
plt.savefig(f"./vary-optim/plots/{optimiser}.png")
plt.show()
