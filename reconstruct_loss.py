import pickle
import matplotlib.pyplot as plt

optimiser = "SSIML1"

TRAIN_LOSS = pickle.load(open(f"./vary-loss/pickles/train_loss_{optimiser}.pkl",
                              'rb'))
VAL_LOSS = pickle.load(open(f"./vary-loss/pickles/val_loss_{optimiser}.pkl", 'rb'))

plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM+L1')
# plt.ylim(0,max(TRAIN_LOSS))
plt.title(f'Using {optimiser} Loss function')
plt.plot(TRAIN_LOSS,'b-')
plt.plot(VAL_LOSS,'r-')
plt.savefig(f"./vary-loss/plots/{optimiser}.png")
plt.show()
