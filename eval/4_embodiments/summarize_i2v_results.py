import os
import pandas as pd
import argparse

def round_dataframe(df, decimals=3):
    float_cols = df.select_dtypes(include=["float", "float64"]).columns
    df[float_cols] = df[float_cols].round(decimals)
    return df


def merge_robot_scores_single(base_dir, robot_types, output_file, prefix_tag):

    all_means = []

    for robot in robot_types:
        csv_path = os.path.join(base_dir, robot, f"score_summary_penalized_{prefix_tag}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} 不存在，跳过。")
            continue

        df = pd.read_csv(csv_path)
        mean_row = df.tail(1).copy()
        if "name" in mean_row.columns:
            mean_row = mean_row.drop(columns=["name"])
        mean_row.insert(0, "Robot_Type", robot)
        all_means.append(mean_row)

    if not all_means:
        print(f"❌ 没有找到任何可用的 {prefix_tag} CSV 文件。跳过输出。")
        return

    merged_df = pd.concat(all_means, ignore_index=True)

    numeric_cols = merged_df.select_dtypes(include="number").columns
    total_mean = merged_df[numeric_cols].mean()
    total_mean_row = pd.DataFrame([{"Robot_Type": "TOTAL_MEAN", **total_mean.to_dict()}])
    merged_df = pd.concat([merged_df, total_mean_row], ignore_index=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df = round_dataframe(merged_df, 3)
    merged_df.to_csv(output_file, index=False)

    print(f"🎯 {prefix_tag.upper()} 汇总结果已保存到: {output_file}\n")


def merge_robot_scores(base_dir, robot_types):

    output_gpt = os.path.join(base_dir, "score_summary_gpt.csv")
    output_qwen = os.path.join(base_dir, "score_summary_qwen.csv")

    merge_robot_scores_single(base_dir, robot_types, output_gpt, "gpt")
    merge_robot_scores_single(base_dir, robot_types, output_qwen, "qwen")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇总多机器人类型 (gpt/qwen) 得分。")
    parser.add_argument("--i2v_model_name", type=str, required=True,
                        help="I2V 模型名称，例如 cosmos2_5-post-trained")
    parser.add_argument("--robot_types", nargs="+", default=["dual_arm", "humanoid", "single_arm", "quad"],
                        help="要汇总的机器人类型列表")
    args = parser.parse_args()

    base_dir = os.path.join("results/4_embodiments", args.i2v_model_name)

    print(f"\n📊 开始汇总 {args.i2v_model_name} 的 {len(args.robot_types)} 类机器人结果 (GPT + QWEN)...\n")
    merge_robot_scores(base_dir, args.robot_types)
