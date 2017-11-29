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


    # Varying the number of observations between 25M and 100M
    for num_obs in [100000000, 75000000, 50000000, 25000000]:
            
        # Number of dimensions will be fixed in 5
        num_dims = 5

        data_path = 'class-data.npz'
        make_data(data_path, num_obs, num_dims, 1826273)

        # Varying the number of K between 2 and 15
        for K in np.arange(3, 16, 3):

            # Varying number of GPUs between 2 and 8, 2 by 2
            for num_gpus in [8, 4, 2, 1]:

                # Varying methods between distribuitedFuzzyCMeans and distribuitedKMeans
                for method in ['distributedKMeans', 'distributedFuzzyCMeans']:

                    # Name of log file, inlcuding the number of obs, dims, K, GPUs and method
                    log_filename =  method + '-GPUs' + str(num_gpus) + '-n_obs' + str(num_obs) + '-n_dims' + str(num_dims) + '-K' + str(K) + '.log'

                    # nvprof process comand to run
                    process_command =   'nvprof --log-file nvprof_logs/' + log_filename + ' python distribuitedClustering.py --n_obs=' + str(num_obs) + ' --n_dim=' + str(num_dims) + ' --K=' + str(K) + ' --n_GPUs=' + str(num_gpus) + ' --n_max_iters=20 --seed=123128 --log_file=executions_log.csv --method_name=' + method + ' --data_file=' + data_path

                    # Trigger new process with the command defined above
                    process = Popen(process_command, shell=True)
                    output = process.communicate()[0]
                    K_error = process.returncode

                    # Print filename in case of errors
                    print(log_filename + ' - Return code: ' + str(K_error))

