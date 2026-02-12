import os
import cv2
import json
import base64
import argparse
import csv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from openai import OpenAI
class Video_preprocess():
    def __init__(self):
        pass
   
    def extract_frames(self, video_path, num_frames=16):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        current_index = 0
        target_set = set(frame_indices)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in target_set:
                frames.append(frame)
                if len(frames) >= num_frames:
                    break
            current_index += 1

        cap.release()
        return frames

    def rgb_to_yuv(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return yuv_frame

    def frames_to_video(self, frames, output_path, fps=8):
        yuv_frames = [self.rgb_to_yuv(frame) for frame in frames]
        video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
        write_video(output_path, video_tensor, fps, video_codec='h264', options={'crf': '18'})

    def convert_video(self, input_path, output_path, num_frames):
        frames = self.extract_frames(input_path,num_frames=num_frames)
        self.frames_to_video(frames, output_path)
    
    def merge_grid(self, image_list, rows=3, cols=2):
        """
        将 image_list 中的帧拼接成 rows × cols 的网格图
        假设 len(image_list) == rows * cols
        """
        assert len(image_list) == rows * cols, f"需要 {rows*cols} 张图片，但传入 {len(image_list)} 张"

        # 每行拼接
        row_images = []
        for r in range(rows):
            row = np.concatenate(image_list[r*cols:(r+1)*cols], axis=1)
            row_images.append(row)

        # 再把行拼接起来
        grid = np.concatenate(row_images, axis=0)
        return grid
    def read_video_path(self, video_path):
        if os.path.isdir(video_path):  # if video_path is a list of videos
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):  # else if video_path is a single video
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path
         

    def convert_video_to_grid(self, video_path,num_image=6):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to image grid with {num_image} frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), f"image_grid_{num_image}frame")
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(0, len(frames) - 1, num_image, dtype=int) #take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f'{vid_id}.jpg')
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path


def create_question_chain(prompt: str) -> str:
    """Construct a step-by-step verification prompt for video generation evaluation."""
    return f"""
    **Task Description**  
    You are an expert in video generation evaluation.  
    The text prompt for the video generation model is **{prompt}**, and the domain is **visual reasoning in manipulation video generation**.  
    Please strictly follow the requirements and examples below to generate a "step-by-step verification" question chain. Evaluate whether the video perfectly implements the text prompt.

    **Question Generation Rules**  
    1. Each question must be based on clearly observable phenomena and allow for binary judgment (yes/no).  
    2. Questions should cover: trigger → feedback → result, and be output in sequence.    

    **Example** 
    [Input Prompt]: 
    "The robot arranges the blocks on the table in the order of green, red, blue, and yellow from left to right from its own perspective."
    
    [Output]:
    1. Does the robot pick up the green block first?  
    2. Is the green block placed on the table on the far left?  
    3. Does the robot pick up the red block next and place it to the right of the green block?  
    4. Does the blue block follow the red block?  
    5. Is the yellow block placed to the rightmost of the sequence?  

    **Generate Output**  
    Now, for the given target text, generate a set of evaluation questions (3-5 questions).  
    """


def create_prompt(view: str,
                  description: str,
                  questions: str,
                  robotic_manipulator: str,
                  manipulated_object: str) -> str:
    """
    生成 VLM 评估提示词（多事件完成度版本）

    view         —— 视频视角，如 'third-person'
    description  —— 视频总体描述
    event_list   —— 一个字符串列表，逐条列出应当发生的事件
    """
    # 把事件列表格式化为编号段落，便于模型逐条检查

    return f'''
You are shown a single image that is a 3 × 2 grid of chronologically ordered frames (read row-wise).
These frames are extracted from an AI-generated video recorded from a {view} perspective.
Video content: {description}  
Robotic manipulator: {robotic_manipulator}  
Manipulated object: {manipulated_object}

Your evaluation focus is: **visual reasoning**.

Please evaluate the video from the following five aspects.  
Each aspect receives a score from **1 to 5**:
- If the aspect is judged as "No", assign **1 point**.  
- If "Yes", assign **2–5 points** depending on quality (5 = perfect).  
- If any aspect in Category A (Action Execution and Visual Reasoning) equals 1, the total score = 1.
- Otherwise, compute the mean of all five scores as the final score.
- BE STRICT WHEN SCORING — if any issue or imperfection is detected, assign 1 or 2 points decisively.

---

### Category A — Action Execution and Visual Reasoning
These aspects focus on how effectively and coherently the robot executes behaviors and whether the visual reasoning process aligns with the described goals.
1) **Action Effectiveness**  
  - Check whether the robot’s motion is physically reasonable (e.g., proper gripper closure, contact location, trajectory smoothness).  
  - Reference scoring:  
    1 = Motion discontinuous, incomplete, or physically implausible.  
    2 = Basically correct and understandable motion.  
    3 = Generally reasonable with slight inaccuracy.  
    4 = Smooth motion and natural contact.  
    5 = Fully consistent with physical and logical principles.

2) **Visual Reasoning Accuracy** 
  Given the following visual reasoning questions, please determine whether each question has been accurately completed, calculate a completion rate, and ensure the evaluation of each question's completion is extremely strict.
  - Visual Reasoning Questions: {questions}
  - Collect ALL questions that are NOT completed into an array called missing_events. Excluding missed questions and obtain the completed_questions. 
  - BE STRICT WHEN JUDGING WHETHER THE QUESTION IS COMPLETED
  - Reference scoring:  
    score = 5 * (completed_questions ÷ total_questions), rounded to 1 decimals.

---

### Category B — Visual and Physical Consistency
These aspects evaluate whether the visual and physical properties of the robot and objects remain stable and realistic throughout the video.
3) **Manipulated Object Consistency**  
   - manipulated object: {manipulated_object}
   - Check whether the manipulated object maintains a consistent shape, structure, and outline over time.
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
     1 = Noticeable changes in appearance such as color, shape, or material; consistency not maintained.
     2 = Moderate differences but still consistent.  
     3 = Minor deformation; overall consistency maintained.  
     4 = Very small jitter or local artifacts only.  
     5 = Completely stable and perfectly consistent appearance.

4) **Robotic Manipulator Consistency**  
   - Robotic Manipulator: {robotic_manipulator} 
   - Check whether the robotic entity, arm or gripper maintains stable geometry and articulation without disconnection or self-intersection.  
   - Note: Evaluate this aspect by comparing all frames to the first frame.
   - Reference scoring:  
     1 = Changes in the form, structure, or appearance of the manipulator or its subcomponents; consistency not maintained.  
     2 = Moderate differences but still consistent.  
     3 = Minor deformation; overall consistency maintained.  
     4 = Very small jitter or local artifacts only.  
     5 = Completely stable and perfectly consistent appearance.

5) **Anomaly Check**  
   - Examine whether any of the following issues are avoided:  
        a) Floating or penetration (violation of physical grounding).  
        b) Objects or robot parts suddenly appear or disappear between frames.  
        c) Non-contact attachment or false grasp (object sticks to gripper without visible closure).  
   - Reference scoring:  
     1 = Occurrence of any of the above anomalies.  
     2 = No obvious anomalies but with noticeable artifacts or noise.  
     3 = Slight noise affecting visual quality.  
     4 = Only tiny visual imperfections.  
     5 = No above anomalies.

---

**[Final Scoring Rule]**

If any aspect in Category A (Action Execution and Visual Reasoning) equals 1, the total score = 1.
Otherwise, compute the mean of all five scores as the final score.

**[Output Format (Strict JSON, No Other Text)]**  
Each aspect and the total must include the following fields:  
- "reason": a brief justification for the score  
- "score": score

Example JSON structure:
{{
    "action_execution": {{"reason": "...", "score": 2}},
    "visual_reasoning_accuracy": {{"reason": "...", "score": 2}},
    "object_consistency": {{"reason": "...", "score": 3}},
    "manipulator_consistency": {{"reason": "...", "score": 3}},
    "anomaly_check": {{"reason": "...", "score": 2}},
    "total": {{"reason": "...", "score": 2.4}}
}}
    '''

def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    return json.loads(json_part)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def compute_final_score(resp: dict) -> float:

    if not isinstance(resp, dict):
        raise TypeError("response 必须是 dict 类型")

    score1 = resp["action_execution"]["score"]
    score2 = resp["visual_reasoning_accuracy"]["score"]
    total_score = resp["total"]["score"]

    a_mean = (score1 + score2) / 2

    final_score = max(min(a_mean, total_score), 1)

    return final_score

def save_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'score', 'prompt', 'details']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            resp = result.get('response', {})

            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    resp = {}
            try:
                score = compute_final_score(resp)
            except Exception:
                score = -1
                    
            writer.writerow({
                'name': result.get('name', ''),
                'score': score,
                'prompt': result.get('prompt', ''),
                'details': resp
            })

def process_single_image_gpt(args_tuple):
    grid_image_name, prompts, image_grid_path, api_key = args_tuple
    client = OpenAI(api_key=api_key)
    try:
        image_index = int(grid_image_name[0:4]) - 1
        prompt_info = prompts[image_index]
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        
        Q_chain = create_question_chain(prompt_info["prompt"])

        response_chain = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q_chain}
                ]
            }],
            max_completion_tokens=8000,
            seed=2026,
        ).choices[0].message.content.strip()
        
        Q = create_prompt(
                          prompt_info["view"], 
                          prompt_info["prompt"],
                          response_chain,
                          prompt_info["robotic manipulator"],
                          prompt_info["manipulated object"]
                          )

        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}
                ]
            }],
            max_completion_tokens=8000,
            seed=2026,
        ).choices[0].message.content.strip()

        try:
            cleaned_output = extract_json(response)
        except Exception:
            cleaned_output = {"score": -1, "reason": response}
            print("Wrong response format!")

        return {
            'name': prompt_info["name"],
            'prompt': prompt_info["prompt"],
            'response': cleaned_output
        }
    except Exception as e:
        return {'name': grid_image_name, 'prompt': 'N/A',
                'response': {'score': -1, 'reason': f'Error: {e}'}}

def run_gpt():
    with open(args.read_prompt_file, 'r') as f:
        prompts = json.load(f)

    image_grid_path = args.image_grid_path
    if image_grid_path is None or not os.path.exists(image_grid_path) or not os.listdir(image_grid_path):
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(args.video_path)
    grid_images = sorted([f for f in os.listdir(image_grid_path) if f[0].isdigit()])

    task_args = [(img, prompts, image_grid_path, args.api_key) for img in grid_images]
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image_gpt, task_args), total=len(task_args)))

    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, 'results.csv')
    save_results_to_csv(results, output_csv)

def process_single_image_qwen(args_tuple):
    grid_image_name, prompts, image_grid_path, api_key = args_tuple
    client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key)
    try:
        image_index = int(grid_image_name[0:4]) - 1
        prompt_info = prompts[image_index]
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        
        Q_chain = create_question_chain(prompt_info["prompt"])

        response_chain = client.chat.completions.create(
            model="qwen3-vl-235b-a22b-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q_chain}
                ]
            }],
            max_completion_tokens=800,
            seed=2026,
        ).choices[0].message.content.strip()
        
        Q = create_prompt(
                          prompt_info["view"], 
                          prompt_info["prompt"],
                          response_chain,
                          prompt_info["robotic manipulator"],
                          prompt_info["manipulated object"]
                          )

        response = client.chat.completions.create(
            model="qwen3-vl-235b-a22b-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": Q},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}
                ]
            }],
            max_completion_tokens=800,
            seed=2026,
        ).choices[0].message.content.strip()
        
        try:
            cleaned_output = extract_json(response)
        except Exception:
            cleaned_output = {"score": -1, "reason": response}
            print("Wrong response format!")
        
        return {
            'name': prompt_info["name"],
            'prompt': prompt_info["prompt"],
            'response': cleaned_output
        }
    except Exception as e:
        return {'name': grid_image_name, 'prompt': 'N/A',
                'response': {'score': -1, 'reason': f'Error: {e}'}}

def run_qwen():
    with open(args.read_prompt_file, 'r') as f:
        prompts = json.load(f)

    image_grid_path = args.image_grid_path
    if image_grid_path is None or not os.path.exists(image_grid_path) or not os.listdir(image_grid_path):
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(args.video_path)
    grid_images = sorted([f for f in os.listdir(image_grid_path) if f[0].isdigit()])

    task_args = [(img, prompts, image_grid_path, args.api_key) for img in grid_images]
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image_qwen, task_args), total=len(task_args)))

    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, 'results.csv')
    save_results_to_csv(results, output_csv)

# =================== 启动入口 ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--image_grid_path", type=str)
    parser.add_argument("--read_prompt_file", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_workers", type=int, default=min(2, cpu_count()))
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "qwen"])
    
    args = parser.parse_args()

    if args.model == "gpt":
        run_gpt()
    else:
        run_qwen()
