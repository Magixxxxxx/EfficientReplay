# python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50boxes.json', )" OUTPUT_DIR './output/emm_15+5_boxes'
python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_1e-2'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_18000'
python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('15+5_50imgs_v2.json', )" OUTPUT_DIR './output/emm_15+5_50imgs_v2_1e-3_b8'

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/voc.yaml" DATASETS.TRAIN "('[1, 15]-voc_train2007.json', )" OUTPUT_DIR './output/base_15'

python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN "('emm+5.json', )" OUTPUT_DIR './output/emmix'


python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN  "('emm+5.json', )" OUTPUT_DIR './output/emmix_0.001' SOLVER.BASE_LR 0.001
python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN  "('emm+5.json', )" OUTPUT_DIR './output/emmix_0.01' SOLVER.BASE_LR 0.01
python mytrain.py --num-gpus 4 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN  "('emm+5.json', )" OUTPUT_DIR './output/emmix_0.02' SOLVER.BASE_LR 0.02

python mytrain.py --num-gpus 2 --config-file "myILOD/configs/emm.yaml" DATASETS.TRAIN  "('emm+5.json', )" OUTPUT_DIR './output/mixup' SOLVER.BASE_LR 0.001

python mytrain.py --num-gpus 1 --config-file "myILOD/configs/emm.yaml" SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.001 DATASETS.TRAIN "('emm+5.json', )" OUTPUT_DIR './output/test' 