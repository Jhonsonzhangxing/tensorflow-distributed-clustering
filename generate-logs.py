import numpy as np
from sklearn.datasets import make_classification
import os

from subprocess import Popen, PIPE, STDOUT

def make_data(filepath, n_obs, n_dim, seed):

    try:
        os.remove(filepath)
    except:
        print('file not found')
    finally:
        (X, Y) = make_classification(n_samples            = n_obs    , 
                                     n_features           = n_dim    ,
                                     n_informative        = n_dim    ,
                                     n_redundant          = 0        ,
                                     n_classes            = 2        ,
                                     n_clusters_per_class = 1        ,
                                     shuffle              = True     ,
                                     random_state         = seed      )

        np.savez(filepath, X=X, Y=Y)

    return True


if __name__ == "__main__":

    # Varying methods between distribuitedFuzzyCMeans and distribuitedKMeans
    for method in ['distributedKMeans', 'distributedFuzzyCMeans']:

        # Varying the number of observations between 2^5 and 2^27
        for num_obs in np.flipud(np.arange(16, 28)):
            error = 0 # flag to check if data is too big
        
            # Varying the number of dimensions between 2 and 20
            for num_dims in np.arange(2, 21):

                if (2**num_obs) * num_dims * 8 > (3200000000):
                    break

                # Function to generate data and save in disk as numpy file. .npz files
                # load much faster then generating them everytime.
                data_path = 'class-data.npz'
                make_data(data_path, 2**num_obs, num_dims, 18273)

                # Varying the number of K between 2 and 15
                for K in np.arange(2, 16):
                    K_error = 0

                    # Varying number of GPUs between 2 and 8, 2 by 2
                    for num_gpus in [8, 6, 4, 2]:

                        # Name of log file, inlcuding the number of obs, dims, K, GPUs and method
                        log_filename =  method + '-GPUs' + str(num_gpus) + '-n_obs' + str(2**num_obs) + '-n_dims' + str(num_dims) + '-K' + str(K) + '.log'

                        # nvprof process comand to run
                        process_command =   'nvprof --log-file nvprof_logs/' + log_filename + ' python distribuitedClustering.py --n_obs=' + str(2**(num_obs)) + ' --n_dim=' + str(num_dims) + ' --K=' + str(K) + ' --n_GPUs=' + str(num_gpus) + ' --n_max_iters=20 --seed=123128 --log_file=executions_log.csv --method_name=' + method + ' --data_file=' + data_path

                        # Print filename in case of errors
                        print(log_filename)

                        # Trigger new process with the command defined above
                        process = Popen(process_command, shell=True)
                        output = process.communicate()[0]
                        K_error = process.returncode

                        if process.returncode > 0:
                            break

                    if K_error >= 1:
                        error += K_error
                        break

                if error >= 2:
                    break
