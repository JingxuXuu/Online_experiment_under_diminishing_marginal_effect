
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as st

T = 6000000
noiselevel = 0.25
interval = [0,1]


output_list = []
'''
regret_lst = []
lb_lst = []
ub_lst = []

for T in [1500, 2000,5000,10000,20000,50000,80000]:
    regret= 0
    regret_sample = []
    for i in range(50):
        regret_sample.append(experiment1(T, noiselevel,interval)[1])
        
    regret = np.mean(regret_sample)
    low_CI_bound, high_CI_bound = st.t.interval(0.95, 50 - 1,loc=regret,scale=st.sem(regret_sample))
    regret_lst.append(regret)
    lb_lst.append(low_CI_bound)
    ub_lst.append(high_CI_bound)
    f = open('result_new4.txt', 'a')
    f.write(str(regret_sample)+'\n')
    f.close()
    
x=[1500,2000,5000,10000,20000,50000,80000]
plt.plot(x,regret_lst,color = 'black',marker='o',markerfacecolor = 'white',label='Estimated value')
plt.fill_between(x, lb_lst, ub_lst, alpha=0.5,label='95% confidence interval')
plt.plot()
'''

font = {'size':12}



for i in range(500):
    T= 500000
    output_list.append(experiment1(T, noiselevel,interval)[0])
f = open('result_clt.txt', 'a')
f.write(str(output_list)+'\n')
f.close()


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
plt.hist(output_list,bins=40, rwidth=0.7,density=True)
x=np.arange(-4,4,0.01)
plt.plot(x,normfun(x,np.mean(output_list),np.std(output_list)),label='Density of a normal distribution',color='red')

plt.xlabel('Value taken by the normalized and centralized estimator',font)
plt.ylabel('Normalized frequency',font)
plt.savefig("estimator_asymptotic_distribution.pdf",dpi=6000,bbox_inches = 'tight')
