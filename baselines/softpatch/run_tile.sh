#datapath=../tile
#datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
#'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
#dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

#python bin/run_patchcore.py --gpu 2 --seed 0 --log_group patchcore-0.01 --log_project My_Results tile \
#patch_core -b wideresnet50 -le layer2 -le layer3 --anomaly_scorer_num_nn 1 \
#sampler -p 0.1 approx_greedy_coreset dataset --resize 1000 --imagesize 1000 --batch_size 2 -d cropped  mvtec $datapath

python bin/run_patchcore.py --gpu 2 --seed 0 --save_segmentation_images --log_group patchcore-0.2-1500 --log_project My_Results tile \
patch_core -b wideresnet50 -le layer2 -le layer3 --anomaly_scorer_num_nn 3 --faiss_on_gpu \
sampler -p 0.2 approx_greedy_coreset dataset --resize 1500 --imagesize 1500 --batch_size 4 -d cropped  mvtec ../tile

#--faiss_on_gpu -le layer2