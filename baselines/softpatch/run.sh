datapath=../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python bin/run_patchcore.py --gpu 0 --seed 0 \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath


###################### BTAD
#datapath=../BTAD
#datasets=('01' '02' '03')
#dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
#
#python bin/run_patchcore.py --gpu 2 --seed 0 \
#--log_group BTAD-patchcore-0.01-512*512 --log_project My_Results results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
#--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
#sampler -p 0.01 approx_greedy_coreset dataset --resize 512 --imagesize 512 "${dataset_flags[@]}" btad $datapath
