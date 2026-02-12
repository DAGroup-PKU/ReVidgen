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

        row_images = []
        for r in range(rows):
            row = np.concatenate(image_list[r*cols:(r+1)*cols], axis=1)
            row_images.append(row)

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
    
    def convert_video_to_frames(self, video_path, num_frames=16):
        video, video_path = self.read_video_path(video_path)
        print(f"start converting video to {num_frames} frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            frames_dir = os.path.join(output_path, vid_id)
            os.makedirs(frames_dir, exist_ok=True)
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path,num_frames=num_frames)
            for frame_count,frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f'{vid_id}_{frame_count:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
        print("video frames stored in: ", output_path)
        return output_path
        
    def convert_video_to_standard_video(self, video_path,num_frames):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to video with 16 frames from path:", video_path)
        
        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
        
        for v in video:
            v_mp4 = v.split(".")[0] + ".mp4"
            self.convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"),num_frames)
        print("finish converting from path: ", video_path)
        print("standard video stored in: ", output_path)
        return output_path

    def convert_video_to_grid(self, video_path,num_image=6):
        video, video_path = self.read_video_path(video_path)
        print("start converting video to image grid with 6 frames from path:", video_path)
    
        output_path = os.path.join(os.path.dirname(video_path), f"image_grid_{num_image}frame")
        os.makedirs(output_path, exist_ok=True)
    
        for v in video:
            vid_id = v.split(".")[0]
            vid_path = os.path.join(video_path,v)
            frames = self.extract_frames(vid_path)
            frame_indices = np.linspace(0, len(frames) - 1, num_image, dtype=int) #take 6 from 16 evenly, 1st & last included
            grid = [frames[i] for i in frame_indices]
            grid_image = self.merge_grid(grid)
            grid_filename = os.path.join(output_path, f'{vid_id}.jpeg')
            cv2.imwrite(grid_filename, grid_image)
        print("finish converting from path: ", video_path)
        print("image grid stored in: ", output_path)
        return output_path

def create_prompt(view, description):
    return f'''    
    The provided image presents sequential frames, arranged in a grid, from a {view} perspective AI-generated task video.
    In this AI-generated video, does the robot successfully perform the task: "{description}"? \n \ 
    Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.\n \
    
    ### Special Notes:
    1) Be very cautious when assigning scores of 4 or 5: these scores should only be used when clearly correct, and free of errors. Do not give 4 or 5 lightly.
    2) Use step-by-step reasoning internally to make your selection.
    3) Your output must be a valid JSON object with two fields:
    - "reason": a breif justification for the given score
    - "score": an integer between 1 and 5
    '''

def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    return json.loads(json_part)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def save_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'score', 'prompt', 'reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            score = result['response'].get("score", "")
            reason = result['response'].get("reason", "")

            writer.writerow({
                'name': result['name'],
                'score': score,
                'prompt': result['prompt'],
                'reason': reason
            })

def process_single_image(args_tuple):
    grid_image_name, image_grid_path, prompts, api_key = args_tuple

    client, real_model_name = create_llm_client(args.model, api_key)

    try:
        image_index = int(grid_image_name[0:4]) - 1
        prompt_info = prompts[image_index]
        full_prompt = prompt_info["prompt"]
        image_path = os.path.join(image_grid_path, grid_image_name)
        img_base64 = encode_image(image_path)
        Q = create_prompt(prompt_info["view"], full_prompt)

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

        raw_output = ask_mllm(Q)

        try:
            cleaned_output = extract_json(raw_output)
            if not isinstance(cleaned_output, dict) or "score" not in cleaned_output or "reason" not in cleaned_output:
                raise ValueError("JSON缺少必要字段")
        except Exception:
            cleaned_output = {"score": -1, "reason": raw_output}

        return {
            'name': prompt_info["name"],
            'prompt': prompt_info["prompt"],
            'response': cleaned_output
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
    save_results_to_csv(results, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str, help="视频文件夹路径")
    parser.add_argument("--image_grid_path", type=str, help="图像网格文件夹路径")
    parser.add_argument("--read_prompt_file", type=str, required=True, help="Prompt JSON 文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出 CSV 文件路径")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "qwen"])
    args = parser.parse_args()
    main()
