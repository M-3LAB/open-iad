datapath=../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-lof-overlap-noise-0.1-no_soft_weight_flag --log_project My_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
--softcore_flag --lof_k 6 --threshold 0.15 --weight_method lof --no_soft_weight_flag \
sampler -p 0.1 approx_greedy_coreset dataset --num_workers 6 "${dataset_flags[@]}" --noise 0.1 --overlap mvtec $datapath
#
#python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-lof-nooverlap-noise-0.1-with_soft_weight_flag --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.15 --weight_method lof \
#sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --noise 0.1  mvtec $datapath

python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-nearest-overlap-noise-0.1-with_soft_weight_flag --log_project My_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
--softcore_flag --lof_k 6 --threshold 0.15 --weight_method nearest \
sampler -p 0.1 approx_greedy_coreset dataset --num_workers 6 "${dataset_flags[@]}" --noise 0.1 --overlap mvtec $datapath

#python bin/run_patchcore.py --gpu 1 --seed 0 --log_group softcore-gaussian-overlap-noise-0.1-with_soft_weight_flag --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.15 --weight_method gaussian \
#sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --noise 0.1 --overlap mvtec $datapath