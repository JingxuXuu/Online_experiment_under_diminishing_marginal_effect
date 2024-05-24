import numpy as np



def concave_f(x):
    return -(x-0.6)**2+2

def experiment(T, noiselevel,interval):
    n_count = 0
    i_count = 0
    current_interval = interval
    while n_count + 2*int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2)) <= T/4:
        i_count += 1
        sample_size = int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2))
        
        sample_left = current_interval[0] + (1/3) * (current_interval[1]-current_interval[0])
        sample_right = current_interval[0] + (2/3) * (current_interval[1]-current_interval[0])
        
        
        s1 = np.mean(np.random.normal(0,noiselevel,sample_size))
        
        f1 = concave_f(sample_left) + s1
        
        s2 = np.mean(np.random.normal(0,noiselevel,sample_size))
        
        f2 = concave_f(sample_right) + s2
        
        if f1 <= f2:
            newcurrent = [sample_left, current_interval[1]]
            
        else:
            newcurrent = [current_interval[0], sample_right]
        current_interval = newcurrent
        
        n_count += 2*sample_size
        
    s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
    result_estimator = concave_f((current_interval[0]+current_interval[1])/2) + s3
    return np.sqrt(T)*(result_estimator-2)

def experiment1(T, noiselevel,interval):
    n_count = 0
    i_count = 0
    current_interval = interval
    regret = 0
    while n_count + 2*int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2)) <= T/3:
        i_count += 1
        sample_size = int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2))
        
        sample_left = current_interval[0] + (1/3) * (current_interval[1]-current_interval[0])
        sample_right = current_interval[0] + (2/3) * (current_interval[1]-current_interval[0])
        
        
        s1 = np.mean(np.random.normal(0,noiselevel,sample_size))
        
        f1 = concave_f(sample_left) + s1
        
        regret += (concave_f(0)-concave_f(sample_left))*sample_size
        regret += (concave_f(0)-concave_f(sample_right))*sample_size
        
        s2 = np.mean(np.random.normal(0,noiselevel,sample_size))
        
        f2 = concave_f(sample_right) + s2
        
        if f1 <= f2:
            newcurrent = [sample_left, current_interval[1]]
            
        else:
            newcurrent = [current_interval[0], sample_right]
        current_interval = newcurrent
        
        n_count += 2*sample_size

    if current_interval[0] == 0:
        s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
        result_estimator = concave_f(0)+ s3
    elif current_interval[1] == 1:
        s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
        result_estimator = concave_f(1) + s3
    else:
        x = (current_interval[0]+current_interval[1])/2
        m0 = 2
        j_count = 1
        while n_count + int(np.ceil(m0*j_count**2))< T:
            h = j_count**(-1/2)
            m = int(np.ceil(m0*j_count**2))
            
            g_estimator = 0
            for k in range(m):
                U = np.random.uniform(low=-1,high =1)
                
                x_max = x+h*U
                x_min = x-h*U
                
                noisy_max = concave_f(x+h*U) + np.random.normal(0,noiselevel)
                regret += concave_f(0) - concave_f(x+h*U)
                noisy_min = concave_f(x-h*U) + np.random.normal(0,noiselevel)
                g_estimator += (noisy_max-noisy_min)*U*(5-7*U**2)*(15/4)/(2*h)
                regret += concave_f(0) - concave_f(x-h*U)
                
            g_estimator = g_estimator/m
            
            x = x + (1/j_count)*g_estimator
            
            if x > current_interval[1]:
                x = current_interval[1]
            if x < current_interval[0]:
                x = current_interval[0]
            j_count += 1
            n_count += 2*m
        
    
        s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
        regret += (concave_f(0.7388600786303446)-concave_f(x))*(T-n_count)
        result_estimator = concave_f(x) + s3
    return np.sqrt(T)*(result_estimator-concave_f(0)),regret


def experiment2(T, noiselevel,interval):
    n_count = 0
    i_count = 0
    current_interval = interval
    regret = 0
    current_interval=[0,1]
    
    
    if 1>0:
        x = 1/2
        m0 = 10
        j_count = 1
        while n_count + int(np.ceil(m0*j_count**2)) < 3*T/4:
            h = j_count**(-1/2)
            m = int(np.ceil(m0*j_count**3))
            
            g_estimator = 0
            for k in range(m):
                U = np.random.uniform(low=-1,high =1)
                
                x_max = x+h*U
                x_min = x-h*U
                
                noisy_max = concave_f(x+h*U) + np.random.normal(0,noiselevel)
                regret += concave_f(0.7388600786303446) - concave_f(x+h*U)
                noisy_min =  concave_f(x-h*U) + np.random.normal(0,noiselevel)
                g_estimator += (noisy_max-noisy_min)*U*(5-7*U**2)*(15/4)/(2*h)
                regret += concave_f(0.7388600786303446) - concave_f(x-h*U)
                
            g_estimator = g_estimator/m
            
            x = x + (1/(1*j_count))*g_estimator
            
            if x > current_interval[1]:
                x = current_interval[1]
            if x < current_interval[0]:
                x = current_interval[0]
            j_count += 1
            n_count += 2*m
        
        s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
        regret += (concave_f(0.7388600786303446)-concave_f(x))*(T-n_count)
        result_estimator = concave_f(x) + s3

    return np.sqrt(T)*(result_estimator-concave_f(0.7388600786303446)),regret

T = 6000000
noiselevel = 0.25
interval = [0,1]

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as st
output_list = []
'''
regret_lst = []
lb_lst = []
ub_lst = []

for T in [1500, 2000,5000,10000,20000,50000,80000]:
    regret= 0
    regret_sample = []
    for i in range(50):
        regret_sample.append(experiment2(T, noiselevel,interval)[1])
        
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
    output_list.append(experiment(T, noiselevel,interval)[0])
f = open('result_clt.txt', 'a')
f.write(str(output_list)+'\n')
f.close()


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
plt.hist(output_list,bins=40, rwidth=0.7,density=True)
x=np.arange(-4,4,0.01)
plt.plot(x,normfun(x,np.mean(output_list),np.std(output_list)),label='Density of a normal distribution',color='red')



'''
plt.ylim(ymin=0)
'''
plt.xlabel('Value taken by the normalized and centralized estimator',font)
plt.ylabel('Normalized frequency',font)
plt.savefig("clt_sgd_new_13.pdf",dpi=6000,bbox_inches = 'tight')

