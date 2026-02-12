import argparse
import json
import os
import csv
def main():
    json_path = args.meta_info_path

    if not os.path.exists(json_path):
        print(f"错误：文件 {json_path} 不存在。")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    perceptible_scores_robotic_manipulator = [
        item["perceptible_amplitude_robotic_manipulator"]
        for item in data
        if "perceptible_amplitude_robotic_manipulator" in item and item["perceptible_amplitude_robotic_manipulator"] is not None
    ]

    perceptible_scores_manipulated_object = [
        item["perceptible_amplitude_manipulated_object"]
        for item in data
        if "perceptible_amplitude_manipulated_object" in item and item["perceptible_amplitude_manipulated_object"] is not None
    ]

    motion_scores = [
        item["motion_smoothness_score"]
        for item in data
        if "motion_smoothness_score" in item and item["motion_smoothness_score"] is not None
    ]

    avg_perceptible_robotic_manipulator = (
        sum(perceptible_scores_robotic_manipulator) / len(perceptible_scores_robotic_manipulator)
        if perceptible_scores_robotic_manipulator else 0
    )
    avg_perceptible_manipulated_object = (
        sum(perceptible_scores_manipulated_object) / len(perceptible_scores_manipulated_object)
        if perceptible_scores_manipulated_object else 0
    )
    avg_motion = (
        sum(motion_scores) / len(motion_scores)
        if motion_scores else 0
    )

    base_dir = os.path.dirname(json_path)
    csv_path = os.path.join(base_dir, "scores_mean.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["score_type", "mean_value"])
        writer.writerow(["perceptible_amplitude_robotic_manipulator", avg_perceptible_robotic_manipulator])
        writer.writerow(["perceptible_amplitude_manipulated_object", avg_perceptible_manipulated_object])
        writer.writerow(["motion_smoothness_score", avg_motion])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 JSON 文件中评分的均值并保存为 CSV")
    parser.add_argument("--meta_info_path", help="result JSON 文件的路径")
    args = parser.parse_args()
    main()