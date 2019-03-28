import numpy as np
from helpers import ptb_iterator

train_data_len = 929589 
valid_data_len = 73760
seq_len = 35
batch_size = 20
num_steps = seq_len
train_data = [i for i in range(train_data_len)]
valid_data = [i for i in range(valid_data_len)]


import time
def run_epoch(model, data, is_train=False, lr=1.0):
    
    epoch_size = ((len(data) // batch_size) - 1) // seq_len
    costs = 0
    iters = 0
    losses = []
    start_time = time.time()
    for step, (x, y) in enumerate(ptb_iterator(data, batch_size, seq_len)):
        loss = 4 + np.random.random() * 0.4
        costs += loss * seq_len
        losses.append(loss)

        if step % (epoch_size // 10) == 10:
                    print('step: '+ str(step) + '\t' 
                        + 'loss (sum over all examples seen this epoch): ' + str(costs) + '\t' \
                        + 'speed (wps):' + str(iters * batch_size / (time.time() - start_time)))
        iters += seq_len

    print('step: '+str(step)+'\t'+"loss: "+str(costs) +'\t')
    print("iters（total_time steps):", iters)
    print("ppl:", np.exp(costs / iters))
    return np.exp(costs / iters), losses


#run_epoch(None, train_data)
run_epoch(None, valid_data)