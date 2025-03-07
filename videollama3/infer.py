import sys
import re
from PIL import Image, ImageDraw
import matplotlib, matplotlib.pyplot as plt
import os

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, load_images
from inference.example_visual_referring import extract_boxes, show_box
from assets.prompts.fps_frompt import FPS_PROMPT


def main():
    disable_torch_init()

    # modal = "text"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": "What is the color of bananas?",
    #     }
    # ]

    # modal = "image"
    # image_path = "./assets/fps/test/SVD持枪穿模.png"
    # frames = load_images(image_path)[0]
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {
    #                 "type": "text",
    #                 "text": "在fps游戏中，这张图片存在什么问题？这个问题出现在图片的哪个位置？请以这种格式回答[[x0,y0,x1,y1]]?",
    #             },
    #         ],
    #     }
    # ]

    modal = "image"
    image_path = "./assets/fps/test/Vector冲锋枪-美杜莎 28020050001 走动换弹时吊坠和武器蛇身穿模.png"
    frames = load_images(image_path)[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": FPS_PROMPT + "这个游戏截图存在什么问题？",
                },
            ],
        }
    ]

    # modal = "video"
    # frames, timestamps = load_video("assets/cat_and_chicken.mp4", fps=1, max_frames=180)
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
    #             {"type": "text", "text": "What is the cat doing?"},
    #         ]
    #     }
    # ]

    model_path = "./work_dirs/videollama3_qwen2.5_2b/stage_3/checkpoint-1"
    model, processor = model_init(model_path)

    inputs = processor(
        images=[frames] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt",
    )

    response = mm_infer(
        inputs, model=model, tokenizer=processor.tokenizer, do_sample=False, modal=modal
    )
    print(response)

    extracted_boxs = extract_boxes(response)
    if extracted_boxs.__len__() > 0:
        show_box(extracted_boxs, frames, image_path, "./assets/saved/")
    else:
        print("No box found")


if __name__ == "__main__":
    main()
