##################### BTAD
datapath=../BTAD
#datasets=('01' '02' '03')
datasets=('01' '02')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

#python bin/run_patchcore.py --gpu 2 --seed 0 \
#--log_group BTAD-patchcore-0.01-512*512--noise-0.1-overlap --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
#--anomaly_scorer_num_nn 1 --patchsize 3 \
#sampler -p 0.01 approx_greedy_coreset dataset --noise 0.1 --overlap --batch_size 8 --resize 512 --imagesize 512 "${dataset_flags[@]}" btad $datapath
#
#python bin/run_patchcore.py --gpu 2 --seed 0 \
#--log_group BTAD-softcore-0.01-512*512--noise-0.1 --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.15 \
#sampler -p 0.01 approx_greedy_coreset dataset --noise 0.1 --batch_size 8 --resize 512 --imagesize 512 "${dataset_flags[@]}" btad $datapath
#
#python bin/run_patchcore.py --gpu 2 --seed 0 \
#--log_group BTAD-softcore-0.01-512*512--noise-0.1-overlap --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.15 \
#sampler -p 0.01 approx_greedy_coreset dataset --noise 0.1 --overlap --batch_size 8 --resize 512 --imagesize 512 "${dataset_flags[@]}" btad $datapath

python bin/run_patchcore.py --gpu 1 --seed 1 \
--log_group BTAD-softcore-0.01-512*512 --log_project My_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
--softcore_flag --lof_k 6 --threshold 0.15 \
sampler -p 0.1 approx_greedy_coreset dataset --batch_size 8 --resize 512 --imagesize 512 "${dataset_flags[@]}"  btad $datapath