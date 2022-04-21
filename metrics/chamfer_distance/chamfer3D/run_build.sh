job_name=build
gpus=1

srun -p dsta --mpi=pmi2 --gres=gpu:${gpus} -n1 \
    --ntasks-per-node=1 --job-name=${job_name}$\
    --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-38 \
    python setup.py install
