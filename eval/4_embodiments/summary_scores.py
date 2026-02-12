import os
import argparse
import pandas as pd


TARGET_COLS = ["Task_Completion", "Visual_Quality"]

ROBOT_TYPES = ["dual_arm", "humanoid", "single_arm", "quad"]


def round_dataframe(df, decimals=3):
    float_cols = df.select_dtypes(include=["float", "float64"]).columns
    df[float_cols] = df[float_cols].round(decimals)
    return df


def load_model_scores(model_dir, model_name, prefix_tag):
    csv_path = os.path.join(model_dir, f"score_summary_{prefix_tag}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ {model_name} 缺少 {os.path.basename(csv_path)}，跳过。")
        return None

    df = pd.read_csv(csv_path)

    if "Robot_Type" not in df.columns:
        print(f"⚠️ {model_name} 的 {csv_path} 缺少 Robot_Type 列，跳过。")
        return None

    result = {"model": model_name}

    # =========================
    # overall_mean
    # =========================
    overall_values = []
    for _, row in df.iterrows():
        if row["Robot_Type"] == "TOTAL_MEAN":
            continue
        vals = [row[c] for c in TARGET_COLS if c in df.columns]
        overall_values.extend(vals)

    if len(overall_values) == 0:
        print(f"⚠️ {model_name} ({prefix_tag}) 无有效 Task_Completion/Visual_Quality，跳过。")
        return None

    result["overall_mean"] = sum(overall_values) / len(overall_values)

    # =========================
    # robot type mean
    # =========================
    for robot in ROBOT_TYPES:
        robot_row = df[df["Robot_Type"] == robot]
        if robot_row.empty:
            result[robot] = None
            continue

        values = [robot_row.iloc[0][c] for c in TARGET_COLS if c in df.columns]
        result[robot] = sum(values) / len(values) if values else None

    return result


def summarize_all_models(ROOT_DIR, prefix_tag):

    all_results = []

    models = [
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ]

    for model_name in models:
        model_dir = os.path.join(ROOT_DIR, model_name)
        res = load_model_scores(model_dir, model_name, prefix_tag)
        if res:
            all_results.append(res)

    if not all_results:
        print(f"❌ {prefix_tag} 未找到任何可汇总的模型结果。\n")
        return

    df = pd.DataFrame(all_results)

    out_csv = os.path.join(ROOT_DIR, f"all_models_summary_{prefix_tag}.csv")
    df = round_dataframe(df, 3)
    df.to_csv(out_csv, index=False)

    print(f"\n🎉 {prefix_tag.upper()} 全模型汇总完成: {out_csv}")
    print(df)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="汇总全部 I2V 模型的得分（支持 gpt + qwen）")
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    print(f"📌 开始汇总 I2V 模型结果")

    summarize_all_models(ROOT_DIR, "gpt")
    summarize_all_models(ROOT_DIR, "qwen")

if __name__ == "__main__":
    main()
