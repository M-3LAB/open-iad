datapath=../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

for noise in {0.1, 0.15}
do
python bin/run_softpatch.py --gpu 0 --seed 0 --log_group softcore-assignment-nearest-denoised-noise"${noise}" --log_project 9-12 results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
--softcore_flag --no_soft_weight_flag \
sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --num_workers 0 \
--noise "${noise}" mvtec ../MVTec
done


