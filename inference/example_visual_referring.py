import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw
from IPython.display import Markdown, display
import matplotlib, matplotlib.pyplot as plt
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.fps_frompt import FPS_PROMPT


# matplotlib.use("TkAgg")


def extract_issue_content(text):
    """
    从文本中提取"发现XXX！"格式的内容，返回XXX部分
    格式和prompt相关
    """
    pattern = r"发现(.*?)！"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def extract_boxes(text):
    def normalize_coordinates(coords):
        """将坐标统一转换为0-1000范围"""
        # 检查坐标是否在0-1范围内
        if all(0 <= x <= 1 for x in coords):
            # 将0-1范围转换为0-1000范围
            return [int(x * 1000) for x in coords]
        # 如果已经是0-1000范围，直接返回
        elif all(0 <= x <= 1000 for x in coords):
            return [int(x) for x in coords]
        else:
            raise ValueError(f"Invalid coordinate range: {coords}")

    # 处理嵌套列表格式 [[x,y,x,y],[x,y,x,y]]
    nested_matches = re.findall(r"\[\[(.*?)\]\]", text)
    if nested_matches:
        result = []
        for match in nested_matches:
            arrays = match.split("],[")
            inner_list = []
            for array in arrays:
                numbers = array.replace("[", "").replace("]", "").split(",")
                coords = [float(num) for num in numbers]
                inner_list.append(normalize_coordinates(coords))
            result.extend(inner_list)
        return result

    # 处理单个列表格式 [x,y,x,y]
    single_matches = re.findall(r"\[([\d.,\s]+)\]", text)
    if single_matches:
        for match in single_matches:
            numbers = match.split(",")
            coords = [float(num) for num in numbers]
            return [normalize_coordinates(coords)]

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


def show_box(issue_content, raw_boxs, image, image_path, save_dir, process=True):
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
        box_coords.extend([str(raw_box_coor) for raw_box_coor in raw_boxs])

    plt.imshow(box_image)
    plt.axis("off")

    # 使用所有box的坐标创建文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"{base_name}_{issue_content}_boxes_{'_'.join(box_coords)}.png"

    plt.savefig(os.path.join(save_dir, output_name), bbox_inches="tight", pad_inches=0)


def predict(model, processor, image_path, conversation):
    try:
        image = Image.open(image_path).convert("RGB")
        saved_dir = "./assets/saved/"
        os.makedirs(saved_dir, exist_ok=True)

        # Single-turn conversation
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=128)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        print(response)
        issue_content = extract_issue_content(response)
        # 依然是相对坐标
        extracted_boxs = extract_boxes(response)
        if extracted_boxs.__len__() > 0:
            # show的是绝对坐标
            show_box(issue_content, extracted_boxs, image, image_path, saved_dir)
        else:
            print("No box found")
    finally:
        # 清理显存
        if "inputs" in locals():
            del inputs
        if "output_ids" in locals():
            del output_ids
        torch.cuda.empty_cache()  # 清理GPU缓存
        import gc

        gc.collect()  # 执行Python的垃圾回收


if __name__ == "__main__":
    # extracted_box = [262, 322, 421, 431]
    # image_path = "./assets/fps/gun_blurry.png"
    # image = Image.open(image_path).convert("RGB")
    # saved_dir = "./assets/saved/"
    # os.makedirs(saved_dir, exist_ok=True)
    # show_box(extracted_box, image, image_path, saved_dir, process=True)

    # NOTE: transformers==4.46.3 is recommended for this script
    # model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
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
    fps_dir = "./scripts/dataset/input/fps_demo/images/"
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
