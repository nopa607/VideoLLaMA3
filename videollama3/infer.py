import sys
import re
from PIL import Image, ImageDraw
import matplotlib, matplotlib.pyplot as plt
import os

sys.path.append("./")
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, load_images


def normalized2raw(box, h, w):
    box = [
        int(box[0] / 1000 * w),
        int(box[1] / 1000 * h),
        int(box[2] / 1000 * w),
        int(box[3] / 1000 * h),
    ]
    return box


def extract_boxes(text):
    matches = re.findall(r"\[\[(.*?)\]\]", text)
    result = []

    for match in matches:
        arrays = match.split("],[")
        inner_list = []

        for array in arrays:
            # Remove any remaining brackets and split by comma to get individual numbers
            numbers = array.replace("[", "").replace("]", "").split(",")
            inner_list.append([int(num) for num in numbers])

        result.extend(inner_list)

    return result


def show_box(raw_box, image, image_path, save_dir, process=True):
    box_image = image.copy()
    # 归一化坐标转换为实际图片中的像素坐标
    if process:
        bbox = normalized2raw(raw_box, box_image.size[1], box_image.size[0])
    else:
        bbox = raw_box
    draw = ImageDraw.Draw(box_image)
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="green", width=2)

    plt.imshow(box_image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"{base_name}_box_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"

    plt.savefig(os.path.join(save_dir, output_name), bbox_inches="tight", pad_inches=0)
    plt.axis("off")
    # plt.show()


def main():
    disable_torch_init()

    # modal = "text"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": "What is the color of bananas?",
    #     }
    # ]

    modal = "image"
    image_path = "./assets/fps/test/SVD持枪穿模.png"
    frames = load_images(image_path)[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "在fps游戏中，这张图片存在什么问题？这个问题出现在图片的哪个位置？请以这种格式回答[[x0,y0,x1,y1]]?",
                },
            ],
        }
    ]

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
                    "text": "在fps游戏中，这张图片存在什么问题？这个问题出现在图片的哪个位置？请以这种格式回答[[x0,y0,x1,y1]]",
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

    extracted_box = extract_boxes(response)[0]
    show_box(extracted_box, frames, image_path, "./assets/saved/")


if __name__ == "__main__":
    main()
