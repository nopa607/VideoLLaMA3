import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw
from IPython.display import Markdown, display
import matplotlib, matplotlib.pyplot as plt
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assets.prompts.fps_frompt import FPS_PROMPT


# matplotlib.use("TkAgg")


def extract_boxes(text):
    # 处理嵌套列表格式 [[x,y,x,y],[x,y,x,y]]
    nested_matches = re.findall(r"\[\[(.*?)\]\]", text)
    if nested_matches:
        result = []
        for match in nested_matches:
            arrays = match.split("],[")
            inner_list = []
            for array in arrays:
                numbers = array.replace("[", "").replace("]", "").split(",")
                inner_list.append([int(num) for num in numbers])
            result.extend(inner_list)
        return result

    # 处理单个列表格式 [x,y,x,y]
    single_matches = re.findall(r"\[([\d,\s]+)\]", text)
    if single_matches:
        for match in single_matches:
            numbers = match.split(",")
            return [[int(num) for num in numbers]]

    return []


def normalized2raw(box, h, w):
    box = [
        int(box[0] / 1000 * w),
        int(box[1] / 1000 * h),
        int(box[2] / 1000 * w),
        int(box[3] / 1000 * h),
    ]
    return box


def raw2normalized(box, h, w):
    box = [
        int(box[0] / w * 1000),
        int(box[1] / h * 1000),
        int(box[2] / w * 1000),
        int(box[3] / h * 1000),
    ]
    box = [min(b, 1000) for b in box]
    return box


def show_box(raw_boxs, image, image_path, save_dir, process=True):
    box_image = image.copy()
    draw = ImageDraw.Draw(box_image)

    # 如果raw_boxs是单个box（一维列表），转换为二维列表
    if not any(isinstance(item, list) for item in raw_boxs):
        raw_boxs = [raw_boxs]

    # 为每个box生成不同的颜色
    colors = ["green", "red", "blue", "yellow", "purple", "orange"]

    # 记录所有box的坐标用于文件名
    box_coords = []

    # 绘制每个边界框
    for i, bbox in enumerate(raw_boxs):
        # 归一化坐标转换为实际图片中的像素坐标
        if process:
            bbox = normalized2raw(bbox, box_image.size[1], box_image.size[0])

        # 使用循环颜色
        color = colors[i % len(colors)]

        # 绘制边界框
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
        box_coords.extend([str(coord) for coord in bbox])

    plt.imshow(box_image)
    plt.axis("off")

    # 使用所有box的坐标创建文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"{base_name}_boxes_{'_'.join(box_coords)}.png"

    plt.savefig(os.path.join(save_dir, output_name), bbox_inches="tight", pad_inches=0)


def predict(model, processor, image_path, conversation):
    image = Image.open(image_path).convert("RGB")
    saved_dir = "./assets/saved/"
    os.makedirs(saved_dir, exist_ok=True)
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()

    # Single-turn conversation
    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)

    extracted_boxs = extract_boxes(response)
    if extracted_boxs.__len__() > 0:
        show_box(extracted_boxs, image, image_path, saved_dir)
    else:
        print("No box found")


if __name__ == "__main__":
    # extracted_box = [262, 322, 421, 431]
    # image_path = "./assets/fps/gun_blurry.png"
    # image = Image.open(image_path).convert("RGB")
    # saved_dir = "./assets/saved/"
    # os.makedirs(saved_dir, exist_ok=True)
    # show_box(extracted_box, image, image_path, saved_dir, process=True)

    # NOTE: transformers==4.46.3 is recommended for this script
    # model_path = "DAMO-NLP-SG/VideoLLaMA3-7B-Image"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 获取fps目录下所有图片文件（不包含子文件夹）
    fps_dir = "./assets/fps/"
    image_files = [
        f
        for f in os.listdir(fps_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.isfile(os.path.join(fps_dir, f))
    ]

    for image_file in image_files:
        image_path = os.path.join(fps_dir, image_file)
        print(f"\nProcessing image: {image_file}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_path}},
                    {
                        "type": "text",
                        "text": FPS_PROMPT + "这个游戏截图存在什么问题？",
                    },
                ],
            }
        ]

        predict(
            model, processor=processor, image_path=image_path, conversation=conversation
        )
