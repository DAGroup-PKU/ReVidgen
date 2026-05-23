import os
import pandas as pd
from tqdm import tqdm
import argparse


def process_model(model_dir, qwen_eval_name="qwen_local"):
    model_name = os.path.basename(model_dir.rstrip("/"))

    task_means_gpt = {}
    task_means_qwen = {}

    all_scores_gpt = []
    all_scores_qwen = []

    for task_name in os.listdir(model_dir):
        task_path = os.path.join(model_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        gpt_csv = os.path.join(task_path, "gpt", "results.csv")
        qwen_csv = os.path.join(task_path, qwen_eval_name, "results.csv")

        df_gpt = None
        df_qwen = None

        if os.path.exists(gpt_csv):
            try:
                df_gpt = pd.read_csv(gpt_csv)
            except:
                df_gpt = None

        if os.path.exists(qwen_csv):
            try:
                df_qwen = pd.read_csv(qwen_csv)
            except:
                df_qwen = None


        if df_gpt is None and df_qwen is None:
            continue

        # GPT scores
        if df_gpt is not None:
            scores_raw = pd.to_numeric(df_gpt["score"], errors="coerce")
            scores_raw = scores_raw[scores_raw >= 0]
            scores = (scores_raw - 1) / 4.0
            task_means_gpt[task_name] = scores.mean()
            all_scores_gpt.extend(scores.tolist())

        # Qwen scores
        if df_qwen is not None:
            scores_raw = pd.to_numeric(df_qwen["score"], errors="coerce")
            scores_raw = scores_raw[scores_raw >= 0]
            scores = (scores_raw - 1) / 4.0
            task_means_qwen[task_name] = scores.mean()
            all_scores_qwen.extend(scores.tolist())

    if not (all_scores_gpt or all_scores_qwen):
        return None

    if all_scores_gpt:
        task_means_gpt["ALL_TASKS_MEAN"] = sum(all_scores_gpt) / len(all_scores_gpt)
    if all_scores_qwen:
        task_means_qwen["ALL_TASKS_MEAN"] = sum(all_scores_qwen) / len(all_scores_qwen)

    if task_means_gpt:
        pd.DataFrame(
            [{"task": k, "mean_score": round(v, 3)} for k, v in task_means_gpt.items()]
        ).to_csv(os.path.join(model_dir, "summary_scores_gpt.csv"), index=False)

    if task_means_qwen:
        pd.DataFrame(
            [{"task": k, "mean_score": round(v, 3)} for k, v in task_means_qwen.items()]
        ).to_csv(os.path.join(model_dir, "summary_scores_qwen.csv"), index=False)

    return model_name, task_means_gpt, task_means_qwen


def main():
    parser = argparse.ArgumentParser(description="汇总多个模型的 GPT 与 QWEN 结果")
    parser.add_argument(
        "--root_dir", type=str, required=True)
    parser.add_argument("--qwen_eval_name", type=str, default="qwen_local")
    args = parser.parse_args()

    root_dir = args.root_dir

    model_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    results = []

    for model_dir in tqdm(model_dirs, desc="Processing i2v models"):
        r = process_model(model_dir, qwen_eval_name=args.qwen_eval_name)
        if r:
            results.append(r)

    if not results:
        print("⚠️ 无可处理模型。")
        return

    all_tasks = sorted(
        {task for _, g, q in results for task in list(g.keys()) + list(q.keys())}
    )

    rows_gpt = []
    rows_qwen = []

    for model_name, gpt_dict, qwen_dict in results:

        # gpt
        row_g = {"model": model_name}
        for task in all_tasks:
            val = gpt_dict.get(task, None)
            row_g[task] = round(val, 3) if isinstance(val, float) else val
        rows_gpt.append(row_g)

        # qwen
        row_q = {"model": model_name}
        for task in all_tasks:
            val = qwen_dict.get(task, None)
            row_q[task] = round(val, 3) if isinstance(val, float) else val
        rows_qwen.append(row_q)

    df_gpt_all = pd.DataFrame(rows_gpt)
    df_qwen_all = pd.DataFrame(rows_qwen)

    df_gpt_all = df_gpt_all.round(3)
    df_qwen_all = df_qwen_all.round(3)

    df_gpt_all.to_csv(
        os.path.join(root_dir, "all_models_summary_gpt.csv"), index=False
    )
    df_qwen_all.to_csv(
        os.path.join(root_dir, "all_models_summary_qwen.csv"), index=False
    )

    print("\n===============================")
    print("📊 GPT 全模型汇总结果")
    print("===============================")
    print(df_gpt_all)

    print("\n===============================")
    print("📊 QWEN 全模型汇总结果")
    print("===============================")
    print(df_qwen_all)

    print("\n✅ 已保存全模型 gpt / qwen 汇总 CSV\n")

if __name__ == "__main__":
    main()
