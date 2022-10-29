datapath=../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

{
python bin/run_patchcore.py --gpu 0 --seed 0 \
--log_group patchcore-overlap-denoise-0.1-thre-0.15 --log_project My_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --noise 0.1 --overlap mvtec $datapath
}&
python bin/run_patchcore.py --gpu 1 --seed 0 \
--log_group patchcore-nooverlap-denoise-0.1-thre-0.15 --log_project My_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" --noise 0.1 mvtec $datapath

wait