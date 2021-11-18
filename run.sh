# python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50boxes.json', )" OUTPUT_DIR './output/emm_15+5_boxes'
python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_1e-2'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_18000'
python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_b8'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/voc.yaml" DATASETS.TRAIN "('[1, 15]-voc_train2007.json', )" OUTPUT_DIR './output/base_15'

