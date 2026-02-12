import os
import argparse
import pandas as pd
import json
import re


def round_dataframe(df, decimals=3):
    float_cols = df.select_dtypes(include=["float", "float64"]).columns
    df[float_cols] = df[float_cols].round(decimals)
    return df

def clean_dataframe(df, name_col="name"):
    df = df.dropna(subset=[name_col])
    df = df[df[name_col].apply(lambda x: bool(re.match(r"^\d+$", str(x))))]
    return df

def sanitize_merged_df(df):


    num_cols = ["PSS", "TAC", "RSS", "MS", "MA"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # bad reply → NaN

    for c in ["Stable_Robo", "Stable_Object"]:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: None if isinstance(x, str) and "bad reply" in x.lower()
                else x
            )

    return df


def pas_penalty(pas, t=0.1, t_low=0.05, delta=0.1):
    if pas < t_low:
        return (t - pas) + delta
    elif pas < t:
        return t - pas
    else:
        return 0.0


def consistent_penalty(robo=None, obj=None):
    penalties = { "B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8 }

    def _get_penalty(x):
        if not isinstance(x, str):
            return 0.0
        val = x.strip().upper()
        base = val[:-1] if (val and val[-1].isdigit()) else val
        return penalties.get(base, 0.0)

    p_robo = _get_penalty(robo)
    p_obj  = _get_penalty(obj)

    if p_robo and p_obj:
        return (p_robo + p_obj) / 2
    else:
        return p_robo


# === Penalized aggregation ===
def filter_and_aggregate(normalized_file, output_file, amplitude_threshold=0.0):
    df = pd.read_csv(normalized_file, dtype={"name": str})

    filtered_df = df[df["MA"] >= amplitude_threshold].copy()
    if filtered_df.empty:
        print("❌ No samples remaining after filtering.")
        return

    # Task Response
    filtered_df["Task_Completion"] = filtered_df[["PSS", "TAC"]].mean(axis=1)

    # Base visual quality
    filtered_df["Visual_Quality_Base"] = (
        filtered_df["RSS"] * 0.8 +
        filtered_df["MS"] * 0.2
    )

    # Penalized visual quality
    def compute_visual_quality(row):
        pas_pen = pas_penalty(row["MA"])
        cons_pen = consistent_penalty(row.get("Stable_Robo"), row.get("Stable_Object"))
        vq = row["Visual_Quality_Base"] - pas_pen - cons_pen
        return max(vq, 0.0)

    filtered_df["Visual_Quality"] = filtered_df.apply(compute_visual_quality, axis=1)

    # Append MEAN row
    last_row = {col: filtered_df[col].mean() for col in [
        "PSS", "TAC", "RSS",
        "MS", "MA",
        "Task_Completion", "Visual_Quality_Base", "Visual_Quality"
    ]}
    last_row["name"] = "MEAN"
    filtered_df = pd.concat([filtered_df, pd.DataFrame([last_row])], ignore_index=True)

    filtered_df = round_dataframe(filtered_df, 3)
    filtered_df.to_csv(output_file, index=False)

def process_single_model(base_dir, prefix_tag, output_root):
    """
    prefix_tag: "gpt" / "qwen"
    output_root: results/model/robot_type/
    """

    if not os.path.exists(base_dir) or len(os.listdir(base_dir)) == 0:
        print(f"⚠️ Skip empty or missing: {base_dir}")
        return

    csv_info = {
        "1_robot_subject_stability/results.csv": "RSS",
        "2_physical_plausibility/results.csv": "PSS",
        "3_task_adherence_consistency/results.csv": "TAC"
    }

    merged_df = None

    for rel_path, col_name in csv_info.items():
        csv_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(csv_path):
            print(f"  ⚠️ Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path, dtype={"name": str})
        if "score" not in df.columns:
            print(f"  ⚠️ 'score' missing in {csv_path}")
            continue

        df["name"] = df["name"].astype(str).apply(lambda x: os.path.splitext(x)[0])

        # Robo & Object
        if "1_robot_subject_stability" in rel_path and "option" in df.columns:
            df["option"] = df["option"].astype(str)
            df["Stable_Robo"] = df["option"].apply(
                lambda x: x.split(",")[0].strip() if "," in x else x.strip()
            )
            df["Stable_Object"] = df["option"].apply(
                lambda x: x.split(",")[1].strip() if "," in x else None
            )

        keep = ["name", "score"]
        if "Stable_Robo" in df.columns: keep.append("Stable_Robo")
        if "Stable_Object" in df.columns: keep.append("Stable_Object")

        df = df[keep].rename(columns={"score": col_name})
        df = clean_dataframe(df)
        merged_df = df if merged_df is None else pd.merge(merged_df, df, on="name", how="outer")

    json_path = os.path.join(output_root, "motion/results.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        json_df = pd.DataFrame([{
            "name": item["index"],
            "MA": item.get("perceptible_amplitude_robotic_manipulator"),
            "MS": item.get("motion_smoothness_score")
        } for item in data])

        json_df = clean_dataframe(json_df)
        merged_df = json_df if merged_df is None else pd.merge(merged_df, json_df, on="name", how="outer")
    else:
        print(f"⚠️ Missing JSON: {json_path}")

    if merged_df is None or merged_df.empty:
        print(f"❌ No valid data for {prefix_tag}.")
        return

    merged_df = merged_df.sort_values(by="name") 

    order = [
        "name",
        "PSS",
        "TAC",
        "RSS",
        "Stable_Robo",
        "Stable_Object",
        "MS",
        "MA"
    ]
    merged_df = merged_df[order]

    merged_df = sanitize_merged_df(merged_df)

    out_csv = os.path.join(output_root, f"score_summary_{prefix_tag}.csv")
    merged_df = round_dataframe(merged_df, 3)
    merged_df.to_csv(out_csv, index=False)

    scaling = {
        "PSS": (1, 5),
        "TAC": (1, 5),
        "RSS": (1, 15),
        "MS": (0, 1),
        "MA": (0, 1)
    }

    normalized_df = merged_df.copy()
    for col, (min_val, max_val) in scaling.items():
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        normalized_df[col] = normalized_df[col].clip(0, 1)

    out_norm = os.path.join(output_root, f"score_summary_normalized_{prefix_tag}.csv")
    normalized_df = round_dataframe(normalized_df, 3)
    normalized_df.to_csv(out_norm, index=False)

    # === penalized ===
    out_pen = os.path.join(output_root, f"score_summary_penalized_{prefix_tag}.csv")
    filter_and_aggregate(out_norm, out_pen, amplitude_threshold=0.0)

# ================================================================
#                         MAIN
# ================================================================
def main():
    output_root = f"results/4_embodiments/{args.i2v_model_name}/{args.robot_type}"

    base_gpt  = os.path.join(output_root, "VQA/gpt")
    base_qwen = os.path.join(output_root, "VQA/qwen")

    process_single_model(base_gpt,  "gpt",  output_root)
    process_single_model(base_qwen, "qwen", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2v_model_name", type=str, default="wanx_i2v_baseline")
    parser.add_argument("--robot_type", type=str, default="humanoid")
    args = parser.parse_args()
    main()
