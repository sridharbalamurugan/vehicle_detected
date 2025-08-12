# evaluate.py (already good)
import torch
from torchvision.ops import box_iou

def evaluate(model, data_loader, device, iou_threshold=0.1):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_precisions = []
    all_recalls = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)

                pred_boxes = output['boxes']
                pred_labels = output['labels']
                scores = output['scores']

                if len(pred_boxes) == 0:
                    all_precisions.append(0.0)
                    all_recalls.append(0.0)
                    total_fn += len(gt_boxes)
                    continue

                keep = scores > 0.1
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                if len(pred_boxes) == 0:
                    all_precisions.append(0.0)
                    all_recalls.append(0.0)
                    total_fn += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)

                matches = []
                for i, pred_label in enumerate(pred_labels):
                    iou_vals = ious[i]
                    matches_for_pred = (iou_vals > iou_threshold) & (gt_labels == pred_label)
                    matches.append(matches_for_pred.any().item())

                tp = sum(matches)
                fp = len(matches) - tp
                fn = max(len(gt_boxes) - tp, 0)


                total_tp += tp
                total_fp += fp
                total_fn += fn

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)

                all_precisions.append(precision)
                all_recalls.append(recall)

    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)

    model.train()

    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_tp + total_fn + 1e-6)
    print(f"Overall Precision: {overall_precision*100:.2f}%")
    print(f"Overall Recall: {overall_recall*100:.2f}%")

    return avg_precision, avg_recall



