import pickle
import matplotlib.pyplot as plt
import pandas as pd

lrs = ['32', '64', '128']

train_losses = []
val_losses = []
df=pd.DataFrame({'32': {'train':[], 'val':[]}, '64': {'train':[], 'val':[]}, '128': {'train':[], 'val':[]} }) 
# '0.1': {'train':[], 'val':[]}, '0.0001': {'train':[], 'val':[]}, '0.00001': {'train':[], 'val':[]} })

df['32']['train'] =  [0.4825,0.4374,0.4293,0.4218,0.418,0.4138,0.4138,0.4106,0.4095,0.407,0.406,0.4029,0.4034,0.4,0.3997,0.3978,0.3976,0.3943,0.3958,0.3946,0.3941,0.3922,0.393,0.3905,0.3908,0.389,0.3897,0.3887,0.3873]
df['32']['val'] = [0.4372,0.4345,0.425,0.4262,0.4124,0.412,0.4084,0.4109,0.4036,0.4039,0.4009,0.4,0.399,0.3982,0.3963,0.3937,0.3932,0.3931,0.3923,0.3909,0.3916,0.394,0.3912,0.3898,0.3901,0.3888,0.3908,0.3878,0.3889]

df['64']['train'] = [0.4484,0.4249,0.4135,0.4084,0.4052,0.4012,0.4005,0.3975,0.3961,0.3953,0.393,0.3918,0.3933,0.3906,0.3892,0.3896,0.3891,0.3869,0.3871,0.3868,0.3862,0.3858,0.3861,0.3837,0.3842,0.3843,0.3838,0.3829,0.3828,0.3823]
df['64']['val'] = [0.4309,0.4123,0.4101,0.4033,0.4012,0.4002,0.3952,0.3949,0.4011,0.3937,0.392,0.3906,0.389,0.3898,0.3895,0.3894,0.3865,0.3857,0.387,0.3875,0.3858,0.3846,0.3867,0.3831,0.3847,0.3846,0.3818,0.3831,0.3824,0.381]

df['128']['train'] = pickle.load(open(f"./vary-ch/pickle/UNET-B3-e30-ch128-ssim-adam-train.pkl",'rb'))
df['128']['val'] = pickle.load(open(f"./vary-ch/pickle/UNET-B3-e30-ch128-ssim-adam-val.pkl", 'rb'))

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
plt.title('Loss against epoch number for large channels')
plt.legend()
# plt.plot(TRAIN_LOSS,'b-')
# plt.plot(VAL_LOSS,'r-')
# plt.savefig(f"./vary-optim/plots/{optimiser}.png")
plt.savefig(f"./vary-ch/plots/all.png")
plt.show()