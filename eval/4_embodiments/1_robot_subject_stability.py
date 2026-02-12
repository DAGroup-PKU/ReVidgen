import os
import cv2
import json
import base64
import argparse
import csv
import numpy as np
import re
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI
import time, random

def create_llm_client(model_name, api_key):
    if model_name.lower() == "gpt":
        return OpenAI(api_key=api_key
        ), "gpt-5-2025-08-07"

    elif model_name.lower() == "qwen":
        return OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key
        ), "qwen3-vl-235b-a22b-instruct"

    else:
        raise ValueError("❌ Unsupported --model, choose from: gpt, qwen")

class Video_preprocess():
    def __init__(self):
        pass
   
    def extract_frames(self, video_path, num_frames=2):
        """
        修改：只取两帧
        - 第一帧 (index=0)
        - 75% 处的帧 (index=int(0.75 * total_frames))
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return frames

        # 确定目标帧索引
        frame_indices = [0, int(0.75 * (total_frames - 1))] if total_frames > 1 else [0]
        target_set = set(frame_indices)

        current_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in target_set:
                frames.append(frame)
                if len(frames) >= len(frame_indices):
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

    def convert_video(self, input_path, output_path):
        frames = self.extract_frames(input_path, num_frames=2)
        self.frames_to_video(frames, output_path)

    def merge_grid(self, image_list, rows=1, cols=2):
        assert len(image_list) == rows * cols, f"需要 {rows*cols} 张图片，但传入 {len(image_list)} 张"

        row_images = []
        for r in range(rows):
            row = np.concatenate(image_list[r*cols:(r+1)*cols], axis=1)
            row_images.append(row)

        grid = np.concatenate(row_images, axis=0)
        return grid
    
    def read_video_path(self, video_path):
        if os.path.isdir(video_path):
            video = os.listdir(video_path)
        elif os.path.isfile(video_path):
            video = [os.path.basename(video_path)]
            video_path = os.path.dirname(video_path)
        video.sort()
        return video, video_path
    
    def convert_video_to_frames(self, video_path):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to 2 frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path, num_frames=2)
            for frame_count, frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f'{vid_id}_{frame_count:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
        print("video frames stored in: ", output_path)
        return output_path
        
    def convert_video_to_standard_video(self, video_path):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 2 frames from path:", video_path)
        
        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
        
        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"))
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to 2-frame image grid from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), f"image_grid_2frame")
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path, v)
            frames = self.extract_frames(vid_path, num_frames=2)
            if len(frames) < 2:
                continue
            grid_image = self.merge_grid(frames, rows=1, cols=2)
            grid_filename = os.path.join(output_path, f'{vid_id}.jpeg')
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path




def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    return json.loads(json_part)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_phrase(phrase_1: str) -> str:
    """
    根据 phrase_1 内容判断是否含有 'gripper(s)' 或 'hand(s)'，
    并返回相应的提示信息 E。
    
    参数:
        phrase_1 (str): 输入短语，如 "robotic gripper" 或 "robotic hands"
    
    返回:
        str: 对应的提示语句
    """
    phrase_1_lower = phrase_1.lower()

    has_gripper = re.search(r'\bgrippers?\b', phrase_1_lower)
    has_arm = re.search(r'\barms?\b', phrase_1_lower)
    has_hand = re.search(r'\bhands?\b', phrase_1_lower)
    has_robot = re.search(r'\brobots?\b', phrase_1_lower)

    if has_gripper or has_arm:
        return f"In the right frame, {phrase_1} is replaced by a 'robotic hand' or 'human hand' when interacting with the object, compared with the left frame."
    elif has_hand:
        return f"In the right frame, {phrase_1} is replaced by a 'human hand' when interacting with the object, compared with the left frame."
    elif has_robot:
        return f"{phrase_1} is missing in the right frame while it exists in the left image."
    else:
        warnings.warn(
            f"Unexpected phrase_1 value: Expected to contain 'gripper(s)', 'hand(s)', or 'robot(s)', but got '{phrase_1}'."
        )
        return ""


def score_mapping(opt_str):
    """根据一个或两个选项映射打分"""
    mapping = {
        # 两个选项组合
        ("A1", "A2"): 15, ("A2", "A1"): 15,
        ("A1", "B2"): 14, ("B1", "A2"): 14, ("A2", "B1"): 14, ("B2", "A1"): 14,
        ("B1", "B2"): 13, ("B2", "B1"): 13,
        ("A1", "C2"): 12, ("C1", "A2"): 12, ("A2", "C1"): 12, ("C2", "A1"): 12,
        ("A1", "D2"): 10, ("D1", "A2"): 10, ("A2", "D1"): 10, ("D2", "A1"): 10,
        ("A1", "E2"): 8,  ("E1", "A2"): 8,  ("A2", "E1"): 8,  ("E2", "A1"): 8,
        ("B1", "C2"): 11, ("C1", "B2"): 11, ("B2", "C1"): 11, ("C2", "B1"): 11,
        ("B1", "D2"): 9,  ("D1", "B2"): 9,  ("B2", "D1"): 9,  ("D2", "B1"): 9,
        ("B1", "E2"): 7,  ("E1", "B2"): 7,  ("B2", "E1"): 7,  ("E2", "B1"): 7,
        ("C1", "C2"): 6,  ("C2", "C1"): 6,
        ("C1", "D2"): 5,  ("D1", "C2"): 5,  ("C2", "D1"): 5,  ("D2", "C1"): 5,
        ("C1", "E2"): 3,  ("E1", "C2"): 3,  ("C2", "E1"): 3,  ("E2", "C1"): 3,
        ("D1", "D2"): 4,  ("D2", "D1"): 4,
        ("D1", "E2"): 2,  ("E1", "D2"): 2,  ("D2", "E1"): 2,  ("E2", "D1"): 2,
        ("E1", "E2"): 1,  ("E2", "E1"): 1,
        # 单一选项（只有 Q1）
        ("A1",): 15,
        ("B1",): 11,
        ("C1",): 7,
        ("D1",): 4,
        ("E1",): 1,
    }

    keys = tuple(opt_str.split(','))
    return mapping.get(keys, "bad reply")

def process_single_image(args_tuple):
    grid_image_name, image_grid_path, prompts, api_key = args_tuple

    client, real_model_name = create_llm_client(args.model, api_key)

    try:
        image_index = int(grid_image_name[0:4]) - 1
        prompt_info = prompts[image_index]
        phrase_1 = prompt_info["robotic manipulator"]
        phrase_2 = prompt_info["manipulated object"]
        full_prompt = prompt_info["prompt"]
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)

        E1 = analyze_phrase(phrase_1)
        phrase_1_lower = phrase_1.lower()
        has_gripper = re.search(r'\bgrippers?\b', phrase_1_lower)
        has_hand = re.search(r'\bhands?\b', phrase_1_lower)
        
        if has_gripper or has_hand:
            Q1 = f''' 
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{phrase_1}' appears in both frames, and evaluate the consistency of '{phrase_1}' between the reference and the generated frame.

Note: 
1) Pay special attention to distinguishing between robotic gripper and robotic hand (if visible). Robotic gripper usually has a small number of rigid gripping jaws or prongs, while a robotic hand has multiple articulated fingers and more complex structures.
2) Changes in orientation or position are acceptable and should not affect the consistency rating.
3) Important: Do NOT assign option A or B lightly. 

Question:
A: '{phrase_1}' in the right frame is clear and consistent with the left image.  
B: '{phrase_1}' in the right frame is mostly consistent with the left image, with minor visual issues.  
C: '{phrase_1}' in the right frame shows noticeable inconsistencies compared with the left image, such as changes in shape, structure.  
D: '{phrase_1}' in the right frame is highly inconsistent with the left image, transforms into another type of '{phrase_1}'.
E: {E1}
The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
'''         
        else:
            Q1 = f''' 
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{phrase_1}' appears in both frames, and evaluate the consistency of '{phrase_1}' between the reference and the generated frame.
Note: 
1) If the subject has a robotic gripper/hand, pay special attention to distinguishing between robotic gripper and robotic hand. Robotic gripper usually has a small number of rigid gripping jaws or prongs, while a robotic hand has multiple articulated fingers and more complex structures.
2) Changes in orientation or position are acceptable and should not affect the consistency rating.
3) Important: Do NOT assign option A or B lightly. 

Question:
A: '{phrase_1}' in the right frame is clear and consistent with the left image.  
B: '{phrase_1}' in the right frame is mostly consistent with the left image, with minor visual issues.  
C: '{phrase_1}' in the right frame shows noticeable inconsistencies compared with the left image, such as changes in shape, structure.  
D: '{phrase_1}' in the right frame is highly inconsistent with the left image, transforms into another type of '{phrase_1}'.
E: {E1}
The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
'''

        Q2 = f'''
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{phrase_2}' appears in both frames, and evaluate the consistency of '{phrase_2}' between the reference and the generated frame.

Note: 
1) Changes in orientation or position are acceptable and should not affect the consistency rating.
2) Important: Do NOT assign option A or B lightly. 

Question: 
A: '{phrase_2}' in the right frame is clear and consistent with the left image.
B: '{phrase_2}' in the right frame is mostly consistent with the left image, with minor visual issues. 
C: '{phrase_2}' in the right frame shows noticeable inconsistencies compared with the left image.  
D: '{phrase_2}' in the right frame undergoes a major transformation, appears as an AI-generated artifact or is duplicated compared with the left image.  
E: '{phrase_2}' is missing in the right frame while it exists in the left image.

The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
'''

        def ask_mllm(question):
            return client.chat.completions.create(
                model=real_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_base64}}
                    ]
                }],
                max_completion_tokens=8000,
                seed=2026,
            ).choices[0].message.content.strip()

        try:
            output_1 = ask_mllm(Q1)
            json_obj_1 = extract_json(output_1)
            if str(phrase_2).strip().lower() == "none":
                phrase_2 = None
            if phrase_2:
                time.sleep(3 + random.random() * 0.5)
                output_2 = ask_mllm(Q2)
                json_obj_2 = extract_json(output_2)
                option_value = f"{json_obj_1['adjust']}1,{json_obj_2['adjust']}2"
                score_tmp = score_mapping(option_value)
                explanation_q2 = json_obj_2.get("explanation", "")
            else:
                option_value = f"{json_obj_1['adjust']}1"
                score_tmp = score_mapping(option_value)
                explanation_q2 = ""
        except Exception as e:
            option_value, score_tmp = "bad reply", "bad reply"
            json_obj_1 = {"explanation": ""}
            explanation_q2 = ""
            print(f"⚠️ reply wrong format at {grid_image_name}: {e}")

        return {
            "name": grid_image_name,
            "prompt": full_prompt,
            "robotic_phrase": phrase_1,
            "object_phrase": phrase_2,
            "option": option_value,
            "score": score_tmp,
            "explanation_q1": json_obj_1.get("explanation", ""),
            "explanation_q2": explanation_q2
        }

    except Exception as e:
        print(f"[Error] {grid_image_name}: {str(e)}")
        return None


def main():
    with open(args.read_prompt_file, 'r') as f:
        prompts = json.load(f)

    image_grid_path = args.image_grid_path
    if image_grid_path is None or not os.path.exists(image_grid_path) or not os.listdir(image_grid_path):
        video_preprocess = Video_preprocess()
        image_grid_path = video_preprocess.convert_video_to_grid(args.video_path)

    grid_images = sorted([f for f in os.listdir(image_grid_path) if f[0].isdigit()])

    task_args = [(img, image_grid_path, prompts, args.api_key) for img in grid_images]

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image, task_args), total=len(task_args)))

    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, 'results.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["name", "prompt", "robotic_phrase", "object_phrase", "option", "score", "explanation_q1", "explanation_q2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if row:
                writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--image_grid_path", type=str)
    parser.add_argument("--read_prompt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "qwen"])
    args = parser.parse_args()
    main()
