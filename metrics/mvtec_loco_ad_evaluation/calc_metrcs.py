import os

def get_metrcs_jsons(dataset_path,tiff_result_path,metric_result_path):
    mvtec_loco_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']
    if(not os.path.exists(metric_result_path)):
        os.makedirs(metric_result_path)
    for object_name in mvtec_loco_list:
        os.system('python3 ./metrics/mvtec_loco_ad_evaluation/evaluate_experiment.py --object_name '+object_name+' --dataset_base_dir '+dataset_path+
            ' --anomaly_maps_dir '+tiff_result_path+' --output_dir '+metric_result_path)

if __name__ == "__main__":
    dataset_path = '/ssd2/m3lab/data/open-ad/mvtecloco/'
    tiff_result_path = '/ssd3/ljq/AD/open-ad/work_dir/fewshot/mvtecloco/_patchcore/1/'
    metric_result_path = '/ssd3/ljq/AD/open-ad/work_dir/fewshot/mvtecloco/_patchcore/1/metrics/'
    get_metrcs_jsons(dataset_path,tiff_result_path,metric_result_path)