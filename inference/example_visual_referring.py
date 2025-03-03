import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw
from IPython.display import Markdown, display
import matplotlib, matplotlib.pyplot as plt
import re
import os

matplotlib.use("TkAgg")


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


def show_box(raw_box, image, image_path, save_dir, process=True):
    box_image = image.copy()
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

    extracted_box = extract_boxes(response)[0]
    show_box(extracted_box, image, image_path, saved_dir)


if __name__ == "__main__":
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

    image_path = "./assets/fps/aimming_no_hand.png"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": image_path}},
                {
                    "type": "text",
                    "text": "在fps游戏中，开镜不应该出现手指对瞄准镜的遮挡，检查一下这张图片是否出现以上问题? 请回答是否有问题，如果有问题，用这种格式回答 [[x0,y0,x1,y1]]，如果没问题，直接回复没有问题",
                },
            ],
        }
    ]
    predict(
        model, processor=processor, image_path=image_path, conversation=conversation
    )

    # image_path = "./assets/fps/gun_1p_model_.jpg"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": {"image_path": image_path}},
    #             {
    #                 "type": "text",
    #                 "text": "在fps游戏中，第一人称手的模型不应该和枪械或者枪械配件有大幅度的模型穿帮，检查一下这张图片是否出现以上问题? 请总结你发现的问题，用这种格式回答 [[x0,y0,x1,y1]]",
    #             },
    #         ],
    #     }
    # ]
    # # predict(processor=processor, image_path=image_path, conversation=conversation)

    # image_path = "./assets/fps/gun_blurry.png"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": {"image_path": image_path}},
    #             {
    #                 "type": "text",
    #                 "text": "在fps游戏中，武器不应该出现纹理模糊，穿模等问题，检查一下这张图片是否出现以上问题? 用这种格式回答 [[x0,y0,x1,y1]]",
    #             },
    #         ],
    #     }
    # ]
    # # predict(processor=processor, image_path=image_path, conversation=conversation)

    # image_path = "./assets/fps/model_miss.png"
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": {"image_path": image_path}},
    #             {
    #                 "type": "text",
    #                 "text": "在fps游戏中，武器不应该出现模型消失等问题，检查一下这张图片是否出现以上问题? 用这种格式回答 [[x0,y0,x1,y1]]",
    #             },
    #         ],
    #     }
    # ]
    # predict(processor=processor, image_path=image_path, conversation=conversation)
