#!/usr/bin/env python
"""
Demo script for surgical instrument segmentation using ISINet (FlowNet2)
Processes images from data/images/ and displays segmentation results
"""

from temp_consistency_module.utils import flow_utils, tools
from temp_consistency_module.models import FlowNet2
import cv2
import torch
import numpy as np
import os
from glob import glob
from os.path import join

import argparse
from collections import namedtuple

import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), 'temp_consistency_module'))


CLASS_NAMES = [
    "__background", "grasper", "bipolar", "needle_driver",
    " scissors", "clip_applier", "suction", "scale"
]


def load_image_pair(img1_path, img2_path, rgb_max=255.):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {img1_path}, {img2_path}")

    # FlowNet2는 일반적으로 RGB 입력을 기대하므로 BGR -> RGB 변환
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1 = img1.astype(np.float32) / rgb_max
    img2 = img2.astype(np.float32) / rgb_max

    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)

    return img1, img2


def main():
    parser = argparse.ArgumentParser(description='ISINet Demo')
    parser.add_argument('--checkpoint', type=str,
                        default='./data/ckp/FlowNet2_checkpoint.pth',
                        help='Path to FlowNet2 checkpoint')
    parser.add_argument('--image_dir', type=str,
                        default='./data/images',
                        help='Directory with input images')
    parser.add_argument('--start_frame', type=int, default=225,
                        help='Start frame number')
    parser.add_argument('--end_frame', type=int, default=241,
                        help='End frame number')
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--div_flow', type=float, default=20.)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fps', type=int, default=5,
                        help='Display FPS')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/demo_isinet',
                        help='Directory to save visualization results')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')

    print("=" * 50)
    print("ISINet Demo - Surgical Instrument Segmentation")
    print("=" * 50)

    # FlowNet2 클래스는 내부적으로 fp16 인자를 사용하므로 명시
    Args = namedtuple('Args', ['rgb_max', 'fp16'])
    model_args = Args(rgb_max=args.rgb_max, fp16=False)

    print("\n[1/4] Building FlowNet2 model...")
    model = FlowNet2(model_args, batchNorm=False, div_flow=args.div_flow)
    model.eval()

    if args.cuda:
        model = model.cuda()

    print("[2/4] Loading checkpoint...")
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   Loaded: {args.checkpoint}")
    else:
        print(f"   Warning: Checkpoint not found at {args.checkpoint}")
        return

    image_files = []
    for frame_num in range(args.start_frame, args.end_frame + 1):
        filepath = join(args.image_dir, f"seq_2_frame{frame_num}.bmp")
        if os.path.exists(filepath):
            image_files.append(filepath)

    if len(image_files) < 2:
        print("Error: Not enough images found!")
        return

    print(f"[3/4] Found {len(image_files)} images")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"   Saving results to: {args.output_dir}")
    print("\n[4/4] Processing...\n")

    window_name = "ISINet Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_image_path = None

    for idx, img_path in enumerate(image_files):
        frame_num = args.start_frame + idx
        current_image_bgr = cv2.imread(img_path)
        if current_image_bgr is None:
            print(f"   Warning: Failed to load {img_path}")
            continue

        if prev_image_path is not None:
            img1, img2 = load_image_pair(
                prev_image_path, img_path, args.rgb_max)

            # 1. 각각 (1, 3, H, W) 텐서 생성
            img1_t = torch.from_numpy(img1).unsqueeze(0).float()
            img2_t = torch.from_numpy(img2).unsqueeze(0).float()

            # 2. 5차원(B, C, N, H, W)으로 합치기
            # 결과 형상: [1, 2, 3, 512, 512] -> permute 후 [1, 3, 2, 512, 512]
            input_data = torch.stack([img1_t, img2_t], dim=1)
            input_data = input_data.permute(0, 2, 1, 3, 4)

            if args.cuda:
                input_data = input_data.cuda()

            with torch.no_grad():
                # FlowNet2 모델은 6채널 입력을 받아 내부에서 3채널씩(이미지 2장) 쪼개어 사용합니다.
                flow_output = model(input_data)

            flow = flow_output[0].cpu().numpy()
            flow = flow.transpose(1, 2, 0)
            flow_vis = flow_utils.flow_to_image(flow)

            # 원본 크기에 맞게 조정 및 출력
            flow_vis = cv2.resize(
                flow_vis, (current_image_bgr.shape[1], current_image_bgr.shape[0]))
            overlay = cv2.addWeighted(current_image_bgr, 0.5, flow_vis, 0.5, 0)

            overlay_path = join(args.output_dir, f"overlay_{frame_num:06d}.png")
            flow_path = join(args.output_dir, f"flow_{frame_num:06d}.png")
            cv2.imwrite(overlay_path, overlay)
            cv2.imwrite(flow_path, flow_vis)

            cv2.imshow(window_name, overlay)
        else:
            first_frame_path = join(
                args.output_dir, f"frame_{frame_num:06d}.png")
            cv2.imwrite(first_frame_path, current_image_bgr)
            cv2.imshow(window_name, current_image_bgr)

        prev_image_path = img_path
        if cv2.waitKey(1000 // args.fps) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\nSaved visualization files under: {args.output_dir}")


if __name__ == '__main__':
    main()
