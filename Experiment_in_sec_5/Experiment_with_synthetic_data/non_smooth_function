import numpy as np


def concave_f(x):
    if x <= 0.5:
      return -0.9*(x-0.5)**2
    else:
      return -1.1*(x-0.5)**2


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
        
        regret += (concave_f(0.5)-concave_f(sample_left))*sample_size
        regret += (concave_f(0.5)-concave_f(sample_right))*sample_size
        
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
                regret += concave_f(0.5) - concave_f(x+h*U)
                noisy_min = concave_f(x-h*U) + np.random.normal(0,noiselevel)
                g_estimator += (noisy_max-noisy_min)*U*(5-7*U**2)*(15/4)/(2*h)
                regret += concave_f(0.5) - concave_f(x-h*U)
                
            g_estimator = g_estimator/m
            
            x = x + (1/j_count)*g_estimator
            
            if x > current_interval[1]:
                x = current_interval[1]
            if x < current_interval[0]:
                x = current_interval[0]
            j_count += 1
            n_count += 2*m
        
    
        s3 = np.mean(np.random.normal(0,noiselevel,T-n_count))
        regret += (concave_f(0.5)-concave_f(x))*(T-n_count)
        result_estimator = concave_f(x) + s3
    return np.sqrt(T)*(result_estimator-concave_f(0.5)),regret

