
import pickle
import matplotlib.pyplot as plt

lr = "0.001"

TRAIN_LOSS = pickle.load(open(f"./vary-lr/pickle/UNET-B14-e30-lr{lr}-ssim-adam-II-train.pkl", 'rb'))
VAL_LOSS = pickle.load(open(f"./vary-lr/pickle/UNET-B14-e30-lr{lr}-ssim-adam-II-val.pkl", 'rb'))

plt.xlabel('Epochs')
plt.ylabel('loss, DSSIM')
# plt.ylim(0,max(TRAIN_LOSS))
plt.title(f'Using {lr} learning rate')
plt.plot(TRAIN_LOSS,'b-')
plt.plot(VAL_LOSS,'r-')
plt.savefig(f"./vary-lr/plots/{lr}-II.png")
plt.show()