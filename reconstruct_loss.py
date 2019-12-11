import pickle
import matplotlib.pyplot as plt

TRAIN_LOSS = pickle.load(open("./vary-optim/pickles/train_loss_adamW.pkl",
                              'rb'))
VAL_LOSS = pickle.load(open("./vary-optim/pickles/val_loss_adamW.pkl", 'rb'))

plt.xlabel('Epochs')
plt.ylabel('loss, ssim')
plt.title('Using AdamW optimiser')
plt.plot(TRAIN_LOSS,'b-')
plt.plot(VAL_LOSS,'r-')
plt.show()
