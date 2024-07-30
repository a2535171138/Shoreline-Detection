import argparse
import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from shapely import LineString, frechet_distance
from uaed_predict import UAED_predict

def draw_shoreline(image, multiline_string):
    height, width = image.shape[:2]

    black_image = np.zeros((height, width), dtype=np.uint8)

    pattern = r"\(\((.*?)\)\)"
    matches = re.findall(pattern, multiline_string)

    for match in matches:
        points = match.split(", ")
        coordinates = []
        for point in points:
            x, y = point.split()
            x = x.strip("()")
            y = y.strip("()")
            coordinates.append((int(float(x)), int(float(y))))

        for i in range(len(coordinates) - 1):
            cv2.line(black_image, coordinates[i], coordinates[i + 1], (255,), 1)

    return black_image
    
def calculate_ODS_metrics(pred_binary, gt_binary, distance_threshold=2.0):
    pred_binary = np.array(pred_binary, dtype=bool)
    gt_binary = np.array(gt_binary, dtype=bool)
    pred_points = np.argwhere(pred_binary)
    gt_points = np.argwhere(gt_binary)

    successful_points = 0

    if len(pred_points) > 0 and len(gt_points) > 0:
        distances = cdist(pred_points, gt_points)
        min_distances = np.min(distances, axis=1)
        min_indices = np.argmin(distances, axis=1)

        matched_gt_points = np.zeros(len(gt_points), dtype=bool)

        for i in range(len(pred_points)):
            while min_distances[i] < distance_threshold:
                matched_gt_index = min_indices[i]

                if not matched_gt_points[matched_gt_index]:
                    successful_points += 1
                    matched_gt_points[matched_gt_index] = True
                    break
                else:
                    distances[i, matched_gt_index] = distance_threshold + 1
                    min_distances[i] = np.min(distances[i])
                    min_indices[i] = np.argmin(distances[i])
            else:
                continue

    if len(gt_points) > 0:
        recall = successful_points / len(gt_points)
    else:
        recall = 0.0

    if len(pred_points) > 0:
        precision = successful_points / len(pred_points)
    else:
        precision = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score


def filter_points_by_distance(pred_points, gt_points, threshold):
    tree = cKDTree(gt_points)
    distances, _ = tree.query(pred_points)
    filtered_points = pred_points[distances <= threshold]
    return filtered_points

def calculate_frechet_similarity(pred_mask, gt_mask, max_distance=1000, threshold=200):
    pred_points = np.argwhere(pred_mask > 0)
    gt_points = np.argwhere(gt_mask > 0)

    if len(pred_points) > 1 and len(gt_points) > 1:
        # Filter pred_points based on distance threshold
        pred_points_filtered = filter_points_by_distance(pred_points, gt_points, threshold)

        if len(pred_points_filtered) > 1:
            pred_line = LineString(pred_points_filtered)
            gt_line = LineString(gt_points)

            frechet_dist = frechet_distance(pred_line, gt_line)
        else:
            frechet_dist = float('inf')
    else:
        frechet_dist = float('inf')

    similarity = max(0, 1 - frechet_dist / max_distance)

    return similarity

def eval(checkpoint_path, input_csv, metric_method, binary_threshold, distance_threshold, save_path):
    df = pd.read_csv(input_csv)
    stats = defaultdict(lambda: defaultdict(lambda: {'num': 0, 'sum': 0, 'mean_metric': 0}))
    overall_stats = {'num': 0, 'sum': 0, 'mean_metric': 0}
    total_time = 0

    for index, row in df.iterrows():
        start_time = time.time()

        input_img = cv2.imread(row['path'], cv2.IMREAD_COLOR)
        pred_mask, _, _ = UAED_predict(input_img, checkpoint_path, binary_threshold)
        gt_mask = draw_shoreline(input_img, row['label'])

        if metric_method == 'ODS':
            precision, recall, metric = calculate_ODS_metrics(pred_mask, gt_mask, distance_threshold)
        elif metric_method == 'frechet_similarity':
            metric = calculate_frechet_similarity(pred_mask, gt_mask)

        overall_stats['num'] += 1
        overall_stats['sum'] += metric
        overall_stats['mean_metric'] = overall_stats['sum'] / overall_stats['num']

        for col in df.columns:
            if col not in ['path', 'label']:
                feature_value = row[col]
                stats[col][feature_value]['num'] += 1
                stats[col][feature_value]['sum'] += metric
                stats[col][feature_value]['mean_metric'] = stats[col][feature_value]['sum'] / stats[col][feature_value]['num']

        iteration_time = time.time() - start_time  # End timer
        total_time += iteration_time

        print(f'{index + 1} -------- current_metric {metric:.3f} -------- mean_metric {overall_stats["mean_metric"]:.3f} -------- Time: {iteration_time:.3f} s')

    print("Final mean metric for each feature and type:")
    for feature in stats:
        for feature_value in stats[feature]:
            print(f'Feature: {feature}, Type: {feature_value}, Mean metric: {stats[feature][feature_value]["mean_metric"]}')
    print(f'Overall Mean metric: {overall_stats["mean_metric"]}')
    average_time = total_time / overall_stats['num']
    print(f'Total average time per iteration: {average_time:.2f} seconds')

    with open(save_path, 'a') as file:
      file.write("Final mean metric for each feature and type:\n")
      for feature in stats:
          for feature_value in stats[feature]:
              file.write(f'Feature: {feature}, Type: {feature_value}, Mean metric: {stats[feature][feature_value]["mean_metric"]}\n')
      file.write(f'Overall Mean metric: {overall_stats["mean_metric"]}\n')
      average_time = total_time / overall_stats['num']
      file.write(f'Total average time per iteration: {average_time:.2f} seconds\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coastline detection and evaluation script")
    parser.add_argument("--binary_threshold", type=int, default=200, help="Binary threshold")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--metric_method", type=str, default='ODS', help="Evaluation metric method")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--distance_threshold", type=int, default=50, help="Distance threshold for evaluation")

    args = parser.parse_args()

    with open(args.save_path, 'a') as file:
      file.write(f'------ Evaluating with checkpoint: {args.model_path} ------\n')
    print(f'------ Evaluating with checkpoint: {args.model_path} ------')
    eval(args.model_path, args.input_csv, args.metric_method, args.binary_threshold, args.distance_threshold, args.save_path)