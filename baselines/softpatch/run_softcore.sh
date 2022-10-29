datapath=../MVTec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

#python bin/run_patchcore.py --gpu 0 --seed 0 --log_group patchcore-noise-0 --log_project rebuttal results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --noise 0  --num_workers 4 mvtec ../MVTec

#python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-lof-noise-0 --log_project rebuttal results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.1 --weight_method lof \
#sampler -p 0.1 approx_greedy_coreset dataset -d capsule -d pill -d screw \
#--noise 0 mvtec ../MVTec

#for seed in $(seq 1 2)
for noise in {0,0.15}
do
python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-assignment-nearest-denoised-noise"${noise}" --log_project 9-12 results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
--softcore_flag --no_soft_weight_flag \
sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --num_workers 0 \
--noise "${noise}" mvtec ../MVTec
done

#python bin/run_patchcore.py --gpu 1 --seed 14 --log_group patchcore-noise0.15-fold2 --log_project find-seed results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" --num_workers 0 \
#--noise 0.15 mvtec ../MVTec


#python bin/run_patchcore.py --gpu 0 --seed 0 --log_group softcore-lof-0.15-noise-0 --log_project 8-29 --save_segmentation_images results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 8 --threshold 0.15 --weight_method lof --no_soft_weight_flag \
#sampler -p 0.1 approx_greedy_coreset dataset "${dataset_flags[@]}" \
#--noise 0 mvtec ../MVTec

#python bin/run_patchcore.py --gpu 1 --seed 1 --log_group softcore-lof-0.15-noise-0.1 --log_project debug results \
#patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --anomaly_scorer_num_nn 1 \
#--softcore_flag --lof_k 6 --threshold 0.5 --weight_method lof --no_soft_weight_flag \
#sampler -p 0.2 approx_greedy_coreset dataset -d cable -d metal_nut -d transistor -d pill \
#--noise 0.1 mvtec ../MVTec

