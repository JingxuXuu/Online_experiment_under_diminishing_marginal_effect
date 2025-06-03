def concave_f(x):
    return 1-np.exp(-1.5*x)+1/2*(1-np.exp(-8*(1-x)))

def experiment1(T, noiselevel,interval):
    n_count = 0
    i_count = 0
    current_interval = interval
    regret = 0
    while n_count + 2*int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2)) <= T/6:
        i_count += 1
        sample_size = int(np.ceil(((3/2)**(4*i_count))*(np.log(T))**2))
        
        sample_left = current_interval[0] + (1/3) * (current_interval[1]-current_interval[0])
        sample_right = current_interval[0] + (2/3) * (current_interval[1]-current_interval[0])
        
        
        
        
        f1 = np.mean((np.random.pareto(3, sample_size) + 1) * 2 * concave_f(sample_left)/3)
        
        regret += (concave_f(0.7388600786303446)-concave_f(sample_left))*sample_size
        regret += (concave_f(0.7388600786303446)-concave_f(sample_right))*sample_size
        
        f2 = np.mean((np.random.pareto(3, sample_size) + 1) * 2 * concave_f(sample_right)/3)
        
        
        
        if f1 <= f2:
            newcurrent = [sample_left, current_interval[1]]
            
        else:
            newcurrent = [current_interval[0], sample_right]
        current_interval = newcurrent
        
        n_count += 2*sample_size

    if current_interval[0] == 0:
        s3 = np.mean(np.random.standard_cauchy(T-n_count))
        result_estimator = concave_f(0)+ s3
    elif current_interval[1] == 1:
        s3 = np.mean(np.random.standard_cauchy(T-n_count))
        result_estimator = concave_f(1) + s3
    else:
        x = (current_interval[0]+current_interval[1])/2
        m0 = 2
        j_count = 1
        while n_count + int(np.ceil(m0*j_count**2))< 5*T/6:
            h = j_count**(-1/2)
            m = int(np.ceil(m0*j_count**2))
            
            g_estimator = 0
            for k in range(m):
                U = np.random.uniform(low=-1,high =1)
                
                x_max = x+h*U
                x_min = x-h*U
                
                noisy_max =(np.random.pareto(3) + 1) *2* concave_f(x+h*U)/3
                regret += concave_f(0.7388600786303446) - concave_f(x+h*U)
                noisy_min = (np.random.pareto(3) + 1) *2* concave_f(x-h*U)/3
                g_estimator += (noisy_max-noisy_min)*U*(5-7*U**2)*(15/4)/(2*h)
                regret += concave_f(0.7388600786303446) - concave_f(x-h*U)
                
            g_estimator = g_estimator/m
            
            x = x + (1/j_count)*g_estimator
            
            if x > current_interval[1]:
                x = current_interval[1]
            if x < current_interval[0]:
                x = current_interval[0]
            j_count += 1
            n_count += 2*m
        
    
        
        regret += (concave_f(0.7388600786303446)-concave_f(x))*(T-n_count)
        result_estimator = np.mean((np.random.pareto(3,T-n_count) + 1) *2* concave_f(1)/3)
    return np.sqrt(T)*(result_estimator-concave_f(0.7388600786303446)), regret
