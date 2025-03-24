import json
import os
import shutil
import torch
import numpy as np
from torchvision.ops import box_iou, nms


def direct_merge(model_files, output_file):
    merged = []

    for file in model_files:
        with open(file, "r") as f:
            data = json.load(f)
            merged.extend(data)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)

    print(f" {len(merged)} results, save to {output_file}")

# IOU
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-10)
    return iou

def iou_merge(model_files, output_file, iou_threshold=0.5):
    predictions = []
    for file in model_files:
        with open(file, "r") as f:
            predictions.extend(json.load(f))

    merged = []
    for pred in predictions:
        keep = True
        for m in merged:
            if pred["image_id"] == m["image_id"] and iou(pred["bbox"], m["bbox"]) > iou_threshold:
                if pred["score"] > m["score"]:
                    m.update(pred)
                keep = False
                break
        if keep:
            merged.append(pred)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"共 {len(merged)} 个检测框，保存至 {output_file}")

# weighted IOU
def weighted_iou_merge(model_files, output_file, iou_threshold=0.5, score_scale=1.2):
    predictions = []
    for i, file in enumerate(model_files):
        with open(file, "r") as f:
            preds = json.load(f)
            # 对第一个模型的分数乘以 score_scale
            if i == 0:
                for pred in preds:
                    pred["score"] *= score_scale
            predictions.extend(preds)

    merged = []
    for pred in predictions:
        keep = True
        for m in merged:
            if pred["image_id"] == m["image_id"] and iou(pred["bbox"], m["bbox"]) > iou_threshold:
                if pred["score"] > m["score"]:
                    m.update(pred)
                keep = False
                break
        if keep:
            merged.append(pred)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"共 {len(merged)} 个检测框，保存至 {output_file}")

# NMS
def nms_merge(model_files, output_file, iou_threshold=0.5):
    predictions = []
    for file in model_files:
        with open(file, "r") as f:
            predictions.extend(json.load(f))

    image_dict = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in image_dict:
            image_dict[image_id] = []
        image_dict[image_id].append(pred)

    merged = []
    for image_id, preds in image_dict.items():
        boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float)
        scores = torch.tensor([p["score"] for p in preds])

        keep = nms(boxes, scores, iou_threshold)

        for idx in keep:
            merged.append(preds[idx])

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"共 {len(merged)} 个检测框，保存至 {output_file}")

# weighted NMS
def weighted_nms_merge(model_files, output_file, iou_threshold=0.5, score_scale=0.3):
    predictions = []
    for i, file in enumerate(model_files):
        with open(file, "r") as f:
            preds = json.load(f)
            # 对第一个模型的分数乘以 score_scale
            if i == 0:
                for pred in preds:
                    pred["score"] *= score_scale
            predictions.extend(preds)

    image_dict = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in image_dict:
            image_dict[image_id] = []
        image_dict[image_id].append(pred)

    merged = []
    for image_id, preds in image_dict.items():
        boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float)
        scores = torch.tensor([p["score"] for p in preds])

        keep = nms(boxes, scores, iou_threshold)

        for idx in keep:
            merged.append(preds[idx])

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"共 {len(merged)} 个检测框，保存至 {output_file}")

# Box Voting
def load_predictions(json_files):
    all_boxes = []
    all_scores = []
    all_labels = []
    image_ids = []

    for file in json_files:
        with open(file, "r") as f:
            predictions = json.load(f)

        boxes, scores, labels, ids = [], [], [], []
        for pred in predictions:
            x, y, w, h = pred["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(pred["score"])
            labels.append(pred["category_id"])
            ids.append(pred["image_id"])

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        image_ids.append(ids)

    return all_boxes, all_scores, all_labels, image_ids

def box_iou(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)

    return iou

def box_voting(json_files, output_file, iou_threshold=0.5):
    all_boxes, all_scores, all_labels, image_ids = load_predictions(json_files)

    # 合并所有模型的预测框
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    merged_ids = []

    for img_idx in range(len(image_ids[0])):
        img_boxes = np.vstack([boxes[img_idx] for boxes in all_boxes])
        img_scores = np.hstack([scores[img_idx] for scores in all_scores])
        img_labels = np.hstack([labels[img_idx] for labels in all_labels])

        sorted_indices = np.argsort(-img_scores)
        img_boxes = img_boxes[sorted_indices]
        img_scores = img_scores[sorted_indices]
        img_labels = img_labels[sorted_indices]

        used = np.zeros(len(img_boxes), dtype=bool)

        for i in range(len(img_boxes)):
            if used[i]:
                continue

            ref_box = img_boxes[i]
            ref_score = img_scores[i]
            ref_label = img_labels[i]

            vote_boxes = []
            vote_scores = []

            for j in range(len(img_boxes)):
                if used[j]:
                    continue

                iou = box_iou(ref_box, img_boxes[j:j + 1])
                if iou >= iou_threshold and ref_label == img_labels[j]:
                    vote_boxes.append(img_boxes[j])
                    vote_scores.append(img_scores[j])
                    used[j] = True

            vote_boxes = np.array(vote_boxes)
            vote_scores = np.array(vote_scores)

            weighted_box = np.sum(vote_boxes.T * vote_scores, axis=1) / np.sum(vote_scores)

            merged_boxes.append(weighted_box)
            merged_scores.append(np.max(vote_scores))
            merged_labels.append(ref_label)
            merged_ids.append(image_ids[0][img_idx])

    results = []
    for i in range(len(merged_boxes)):
        x1, y1, x2, y2 = merged_boxes[i]
        results.append({
            "image_id": merged_ids[i],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(merged_scores[i]),
            "category_id": int(merged_labels[i])
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"共 {len(results)} 个检测框，保存至 {output_file}")

if __name__ == "__main__":
    a_values = [1, 2, 3]
    b_values = [1, 5, 10]
    nms_params = {1: (0.5, 0.4), 5: (0.5, 0.8), 10: (0.1, 1)}

    source_dir_dino = "DINO_result"
    source_dir_glip = "GLIP_result"
    target_dir = "ensemble_results"

    os.makedirs(target_dir, exist_ok=True)

    for a in a_values:
        for b in b_values:
            filename = f"dataset{a}_{b}shot.json"
            source_file_glip = os.path.join(source_dir_glip, filename)
            target_file = os.path.join(target_dir, filename)

            if a in [1, 2]:
                # dataset1 dataset2 直接取 GLIP 的结果
                shutil.copy(source_file_glip, target_file)
            elif a == 3:
                # dataset3 取 GLIP 和 DINO 的 ensemble 结果
                source_file_dino = os.path.join(source_dir_dino, filename)
                param1, param2 = nms_params[b]
                weighted_nms_merge([source_file_glip, source_file_dino], target_file, param1, param2)