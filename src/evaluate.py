# src/evaluate.py
import torch
import torchvision
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)
