import numpy as np

def acceptance_rate(step_cache, burnin):
    return np.count_nonzero(step_cache[burnin:])/step_cache[burnin:].shape[0]

def run_acceptance_rate(step_cache, window, burnin):
    step_cache = step_cache[burnin:]
    run_acc = np.zeros(len(step_cache))
    
    for i in range(len(step_cache)):
        if i==0:
            run_acc[i] = step_cache[0]
            continue
        if i<window:
            run_acc[i] = acceptance_rate(step_cache[0:i], 0)
        else:
            run_acc[i] = acceptance_rate(step_cache[i-window:i], 0)
            
    return run_acc

    