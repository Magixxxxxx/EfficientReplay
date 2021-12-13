import os
import logging
from collections import OrderedDict
from random import randint
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from myILOD.utils.register import my_register
import detectron2.utils.comm as comm

from PIL import Image, ImageDraw
from detectron2.data.detection_utils import convert_PIL_to_numpy
import numpy as np
import torch, sys, random, json, logging, time, cv2

from detectron2.data import build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T
from detectron2.data.build import get_detection_dataset_dicts, DatasetFromList, MapDataset

class myAug(T.augmentation.Augmentation):

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # EMM
        self.memory = self.build_memory(cfg)        

    def CutPaste(self, each_img):
        # each_img: 3 (b, g, r) * H * W
        replay_ann = random.choice(self.memory['annotations'])
        replay_img = self.memory['images'][replay_ann['image_id'] - 1]
        mm = Image.open('datasets/VOC2007/JPEGImages/' + replay_img['file_name'])

        # cut
        x1, y1, x2, y2 = replay_ann['bbox']
        bbox = (x1, y1, x1 + x2, y1 + y2)
        mm_cat = torch.tensor(replay_ann['category_id']) 
        mm_cut = mm.crop(bbox)
        
        # paste
        paste_x = random.randint(0, max(0, each_img['image'].size()[2] - x2))
        paste_y = random.randint(0, max(0, each_img['image'].size()[1] - y2))
        b, g, r = cv2.split(each_img['image'].byte().permute(1, 2, 0).numpy())
        new_img = Image.fromarray(cv2.merge([r, g, b]))
        new_img.paste(mm_cut, (paste_x, paste_y))

        # fix labels
        mm_box = torch.tensor([float(i) for i in [paste_x, paste_y, paste_x + x2, paste_y + y2]])
        gt_boxes = torch.unsqueeze(mm_box, 0)
        gt_classes = torch.unsqueeze(mm_cat, 0)

        for box, cat in zip(each_img['instances']._fields['gt_boxes'].tensor, each_img['instances']._fields['gt_classes']):
            ixmin = np.maximum(mm_box[0], box[0])
            iymin = np.maximum(mm_box[1], box[1])
            ixmax = np.minimum(mm_box[2], box[2])
            iymax = np.minimum(mm_box[3], box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (mm_box[2] - mm_box[0] + 1.0) * (mm_box[3] - mm_box[1] + 1.0)
            overlaps_of_box = inters / uni
            if overlaps_of_box <= 0.5:
                gt_boxes = torch.cat((gt_boxes, torch.unsqueeze(box, 0)))
                gt_classes = torch.cat((gt_classes, torch.unsqueeze(cat, 0)))
        
        # a = ImageDraw.ImageDraw(new_img)
        # for b in gt_boxes:
        #     a.rectangle([int(i) for i in b])
        # new_img.save(str(paste_x) + ".jpg")

        each_img['image'] = torch.as_tensor(np.ascontiguousarray(convert_PIL_to_numpy(new_img, "BGR").transpose(2, 0, 1)))
        each_img['instances']._fields['gt_boxes'].tensor = gt_boxes
        each_img['instances']._fields['gt_classes'] = gt_classes

    def Mixup(self, each_img):

        # input data
        img1 = each_img['image'].byte().permute(1, 2, 0).numpy()
        lambd = np.random.beta(2,2)
        
        # memory data
        mm_data = random.choice(self.memory)
        img2= mm_data['image'].byte().permute(1, 2, 0).numpy()

        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
        mix_img = mix_img.astype('uint8')

        each_img['image'] = torch.as_tensor(np.ascontiguousarray(mix_img.transpose(2, 0, 1)))
        each_img['instances']._fields['gt_boxes'].tensor = torch.cat((each_img['instances']._fields['gt_boxes'].tensor, mm_data['instances']._fields['gt_boxes'].tensor))
        each_img['instances']._fields['gt_classes'] = torch.cat((each_img['instances']._fields['gt_classes'], mm_data['instances']._fields['gt_classes']))

        # drawimg = Image.fromarray(mix_img)
        # a = ImageDraw.ImageDraw(drawimg)
        # for b in each_img['instances']._fields['gt_boxes'].tensor:
        #     a.rectangle([int(i) for i in b])
        # drawimg.save("0.jpg")

    def run_step(self):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)

        # TODO: EMM
        # for each_img in data:
        #     self.Mixup(each_img)
        # END

        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        print(loss_dict)
        sys.exit(0)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ):
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PascalVOCDetectionEvaluator(dataset_name) 

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def build_train_loader(cls, cfg):

        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        augmentation.append(T.RandomFlip())

        mapper = DatasetMapper(cfg, True, augmentations=augmentation)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_memory(cls, cfg):
        memory_dict = get_detection_dataset_dicts(
            cfg.DATASETS.MEMORY,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
            )
        memory_dataset = DatasetFromList(memory_dict)
        memory_dataset = MapDataset(memory_dataset, DatasetMapper(cfg, True))
        return memory_dataset

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() 
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置
    cfg.freeze()

    default_setup(cfg, args)

    return cfg

def main(args):

    # ZJW: Myregister
    my_register()

    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    model = Trainer.build_model(cfg)  
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.dist_url='tcp://127.0.0.1:{}'.format(randint(30000,50000))
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )