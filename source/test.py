import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
barc = pkl.load(open('/home/xlv/Desktop/BaRC/source/runs/PlanarQuad-v0_backreach_02-Oct-2018_21-09-58-star/figures/global_ppo_rewards','rb'))
rand = pkl.load(open('/home/xlv/Desktop/BaRC/source/runs/PlanarQuad-v0_random_04-Oct-2018_13-00-10-star/figures/global_ppo_rewards','rb'))

#fig.show()
barc_data = barc.axes[0].lines[0].get_data()
rand_data = rand.axes[0].lines[0].get_data()

index = barc_data[0]
barc_value = barc_data[1]
rand_value = rand_data[1]

interval = np.arange(0,800,4)
#interval = np.arange(0,40,2)

plt.figure()
plt.plot(index[interval],barc_value[interval],label="BaRC")
plt.plot(index[interval],rand_value[interval],label="Random")
plt.xlabel("PPO iteration")
plt.ylabel("Avg rewards per training step")
plt.legend(loc='upper right')
plt.show()

print("data",barc_data)
'''

#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
plt.ion()
#sns.distplot([0,1,2,3,4,5,6,7,6,5,4,3,2,1,0],rug=True)
fig, ax = plt.subplots(figsize=(5,5),ncols=1,nrows=5,squeeze=False)
for i in range(5):
    plt.cla()
    #plt.plot([i, i+1])
    #sns_plot = sns.distplot([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], rug=True)

    sns.distplot([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], rug=True,ax=ax[i,0])
    print("fuck")
    plt.pause(0.1)
    #sns_plot.figure.savefig("demo.png")

plt.ioff()
plt.show()