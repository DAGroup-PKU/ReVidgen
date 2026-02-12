import argparse
import os
import sys
import math
from tqdm import tqdm

import numpy as np
import json
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


from PIL import ImageDraw, ImageFont
from matplotlib import colormaps

sys.path.append(os.path.join(os.getcwd(), "pkgs", "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "pkgs", "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "pkgs", "Grounded-Segment-Anything", "segment_anything"))
sys.path.append(os.path.join(os.getcwd(), "pkgs", "co-tracker"))
sys.path.append(os.path.join(os.getcwd(), "pkgs", "sam2"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Co-Tracker
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None, None, None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        print("Error reading frames from the video")
        return None, None, None, None

    # take the first frame as the query image
    frame_rgb = frames[0]
    image_pil = Image.fromarray(frame_rgb)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    
    return image_pil, image, frame_rgb, np.stack(frames)


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def calculate_motion_degree(keypoints, video_width, video_height):
    """
    Calculate the normalized motion amplitude for each batch sample
    
    Parameters:
    keypoints: torch.Tensor, shape [batch_size, 49, 792, 2]
    video_width: int, width of the video
    video_height: int, height of the video
    
    Returns:
    motion_amplitudes: torch.Tensor, shape [batch_size], containing the normalized motion amplitude for each batch sample
    """

    # Calculate the length of the video diagonal
    diagonal = torch.sqrt(torch.tensor(video_width**2 + video_height**2, dtype=torch.float32))
    
    # Compute the Euclidean distance between adjacent frames
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # shape [batch_size, 48, 792]
    
    # Normalize the distances (divide by the diagonal length)
    normalized_distances = distances / diagonal
    
    # Sum the normalized distances to get the total normalized motion distance for each keypoint
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # shape [batch_size, 792]
    
    # Compute the normalized motion amplitude for each batch sample (mean of total normalized motion distance for all points)
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)  # shape [batch_size]
    
    return motion_amplitudes

def save_image_with_mask_and_boxes(image_array, mask_tensor, boxes_tensor, save_path):
    """
    Args:
        image_array: numpy array of shape (H, W, 3), RGB
        mask_tensor: torch.Tensor of shape (N, 1, H, W) int (N 可以是 1 或 2)
        boxes_tensor: torch.Tensor of shape (N, 4), in xyxy format
        save_path: str, path to save .png image
    """
    image = image_array.copy()
    num_masks = mask_tensor.shape[0]

    colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HSV, cv2.COLORMAP_OCEAN]

    for i in range(num_masks):
        mask = mask_tensor[i].squeeze(0).cpu().numpy().astype(np.uint8) * 255  # (H, W)
        mask_color = cv2.applyColorMap(mask, colormaps[i % len(colormaps)])
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)

        mask_bool = mask > 0
        image[mask_bool] = (0.5 * image[mask_bool] + 0.5 * mask_color[mask_bool]).astype(np.uint8)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(num_masks):
        box = boxes_tensor[i].int().tolist()
        x1, y1, x2, y2 = box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_bgr, f"object_{i+1}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(save_path, image_bgr)
    print(f"Saved image with {num_masks} boxes and masks to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--target_type", type=str, required=False, default="robotic_manipulator", help="type of object to be inspected")
    parser.add_argument("--meta_info_path", type=str, required=True, help="path to meta info json")
    parser.add_argument("--text_prompt", type=str, required=False, help="text prompt", 
            default="person. dog. cat. horse. car. ball. robot. bird. bicycle. motorcycle. surfboard. skateboard. bucket. bat. basketball. " \
              "racket. kitten. puppy. fish. laptop. umbrella. wheelchair. drone. scooter. rollerblades. truck. bus. skier. snowboard. " \
              "sled. kayak. canoe. sailboat. guitar. piano. drum. violin. trumpet. saxophone. clarinet. flute. accordion. telescope. " \
              "microscope. treadmill. rope. ladder. swing. tugboat. train.")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--grid_size", type=int, default=30, help="Regular grid size")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # model cfg
    config_file = "pkgs/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    grounded_checkpoint = "checkpoints/GroundingDino/groundingdino_swinb_cogcoor.pth"
    bert_base_uncased_path = "checkpoints/BERT/google-bert/bert-base-uncased"
    sam_checkpoint = "./checkpoints/SAM/sam2.1_hiera_large.pt"
    sam_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    cotracker_checkpoint = "checkpoints/Cotracker/scaled_offline.pth"

    meta_info_path = args.meta_info_path
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    grid_size = args.grid_size
    device = args.device

    # load model
    grounding_model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # initialize SAM
    sam_predictor = SAM2ImagePredictor(build_sam2(sam_model_cfg, sam_checkpoint))

    # intialize Co-Tracker
    cotracker_model = CoTrackerPredictor(
        checkpoint=cotracker_checkpoint,
        v2=False,
        offline=True,
        window_len=60,
    ).to(device)

    # load meta info json
    with open(args.meta_info_path, 'r') as f:
        meta_infos = json.load(f)
    
    visualization_dir = os.path.join(os.path.dirname(args.meta_info_path), "visualization", args.target_type)
    os.makedirs(visualization_dir, exist_ok=True)

    for meta_info in tqdm(meta_infos, desc="Motion Degree: Grounded SAM Segmentation"):
        image_pil, image, image_array, video = load_video(meta_info['filepath'])

        text_prompt = meta_info[args.target_type] + '.' # robotic_manipulator / manipulated_object
        if text_prompt.lower() == "none.":
            print(f"skip {args.target_type} because prompt is None in {meta_info['prompt']}")
            metric_name = f'perceptible_amplitude_{args.target_type}'
            meta_info[metric_name] = None
            with open(args.meta_info_path, 'w') as f:
                json.dump(meta_infos, f, indent=4)
            continue 
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            grounding_model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        
        # no detect object
        if boxes_filt.shape[0] == 0:
            print(f"can not detect {text_prompt} in {meta_info['prompt']}")
        else:
            sam_predictor.set_image(image_array)

            # convert boxes into xyxy format
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()

            if args.target_type=="robotic_manipulator":
                num_box = 2 if meta_info['robotic_manipulator'] in ["robotic grippers", "robotic arms", "robotic hands", "humanoid robots"] else 1
            else:
                num_box = 1
            boxes_filt = boxes_filt[:num_box]
            # run sam model
            mask, _, _ = sam_predictor.predict(
                point_coords = None,
                point_labels = None,
                box = boxes_filt.to(device),
                multimask_output = False,
            )
            mask = torch.from_numpy(mask).to(torch.uint8)
            if mask.ndim == 3:  
                # (N,H,W) → (N,1,H,W)
                mask = mask.unsqueeze(1)
           
        video_name = os.path.splitext(os.path.basename(meta_info['filepath']))[0]
        save_path = os.path.join(visualization_dir, f"{video_name}_box_mask.png")
        if boxes_filt.shape[0] != 0:
            save_image_with_mask_and_boxes(image_array, mask, boxes_filt, save_path)


        # load the input video frame by frame
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        video_width, video_height = video.shape[-1], video.shape[-2]
        video = video.to(device)

        if boxes_filt.shape[0] != 0:
            background_mask = torch.any(~mask, dim=0).to(torch.uint8) * 255
        else:
            background_mask = torch.ones((1, video_height, video_width), dtype=torch.uint8, device=device) * 255
        
        # eval background (camera) motion degree
        background_mask = background_mask.unsqueeze(0)
        pred_tracks, pred_visibility = cotracker_model(
            video,
            grid_size=grid_size,
            grid_query_frame=0,
            backward_tracking=True,
            segm_mask=background_mask
        )
        
        background_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
        # meta_info['motion_amplitude']['camera'] = background_motion_degree.item()

        metric_name = f'perceptible_amplitude_{args.target_type}'
        if boxes_filt.shape[0] != 0:
            subject_mask = torch.any(mask, dim=0).to(torch.uint8) * 255
            # eval subject motion degree
            subject_mask = subject_mask.unsqueeze(0)
            pred_tracks, pred_visibility = cotracker_model(
                video,
                grid_size=grid_size,
                grid_query_frame=0,
                backward_tracking=True,
                segm_mask=subject_mask
            )
            
            subject_motion_degree = calculate_motion_degree(pred_tracks, video_width, video_height).item()
            # subject_motion_degree = subject_motion_degree.item()
            if subject_motion_degree > background_motion_degree:
                subject_motion_degree = subject_motion_degree - background_motion_degree
            if not np.isnan(subject_motion_degree):
                meta_info[metric_name] = subject_motion_degree
            else:
                meta_info[metric_name] = background_motion_degree
        else:
            meta_info[metric_name] = background_motion_degree

        # save meta info per video
        with open(args.meta_info_path, 'w') as f:
            json.dump(meta_infos, f, indent=4)