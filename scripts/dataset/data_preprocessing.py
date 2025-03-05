# 数据集处理，从vit格式的数据转化为模型输入格式
# 支持数据集划分，数据集增强，数据总结
import json
import pandas as pd
import os
import random
import shutil
from via_sample import ViaDataSample


class Conversation(object):
    def __init__(self, conv_from: str, value: str):
        self.conv_from = conv_from
        self.value = value

    def to_json(self):
        return {"from": self.conv_from, "value": self.value}


class DataSample(object):
    """
    一个数据样本
    """

    def __init__(self, image: list[str], conversations: list[Conversation]):
        self.image = image
        self.conversations = conversations

    def to_json(self):
        return {
            "image": self.image,
            "conversations": [conv.to_json() for conv in self.conversations],
        }

    def save_json(self, json_path: str = "./scripts/dataset/output/dataSample.json"):
        with open(json_path, "w") as f:
            json.dump(self.to_json(), f, indent=4, ensure_ascii=False)


class DatasetProcessor(object):
    def __init__(self, via_instance: ViaDataSample = None):
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.datasamples: list[DataSample] = []
        self.via_instance = via_instance  # 数据集实例
        self.project_name = os.path.basename(self.via_instance.project_path)
        self.input_path = os.path.join("./scripts/dataset/input/", self.project_name)
        self.output_path = os.path.join("./scripts/dataset/output/", self.project_name)
        os.makedirs(self.output_path, exist_ok=True)
        self.json_path = os.path.join(self.output_path, "data.json")  # via格式化样本
        self.image_path = os.path.join(self.input_path, "images/")  # via对应的图片路径
        self.train_json_path = os.path.join(self.output_path, "train", "train.json")
        self.train_image_path = os.path.join(self.output_path, "train", "images")
        self.val_json_path = os.path.join(self.output_path, "val", "val.json")
        self.val_image_path = os.path.join(self.output_path, "val", "images")
        self.test_json_path = os.path.join(self.output_path, "test", "test.json")
        self.test_image_path = os.path.join(self.output_path, "test", "images")

    def process(self):
        self.parse_csv_to_samples()
        self.export_json()
        self.split_dataset()

    def parse_csv_to_samples(self):
        """
        从原始CSV文件解析数据并转换为DataSample对象列表
        """
        df = pd.read_csv(self.via_instance.csv_path)

        for _, row in df.iterrows():
            if row["region_count"] == 0:
                continue

            # 获取必要信息
            filename = row["filename"]
            bug_desc = self.via_instance.get_bug_description(row)
            bug_type = self.via_instance.get_bug_type(row)
            bbox = self.via_instance.get_bonding_box(row)

            if not all([filename, bug_desc, bug_type, bbox]):
                continue

            # 构建对话
            conversations = [
                Conversation(
                    "human",
                    f"<image>\n在fps游戏中，这张图片存在什么问题？这个问题出现在图片的哪个位置？请给出具体坐标",
                ),
                Conversation(
                    "gpt",
                    f"这张图片显示了一个{bug_type}类型的问题：{bug_desc}.具体坐标是:[{bbox}]",
                ),
            ]

            # 创建数据样本并添加到列表
            data_sample = DataSample([filename], conversations)
            self.datasamples.append(data_sample)

    def export_json(self):
        """
        DataSamples转换为json格式
        """
        data_samples_json = [data_sample.to_json() for data_sample in self.datasamples]
        with open(self.json_path, "w") as f:
            json.dump(data_samples_json, f, indent=4, ensure_ascii=False)
        return data_samples_json

    def split_dataset(self):
        """
        数据集切分为训练集、验证集和测试集
        """
        if not os.path.exists(self.json_path):
            raise Exception("请先执行process方法")

        # 读取原始数据集
        with open(self.json_path, "r") as f:
            data = json.load(f)

        # 计算数据集大小
        total_size = len(data)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)

        random.shuffle(data)

        # 划分数据集
        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]

        # 创建输出目录
        os.makedirs(os.path.dirname(self.train_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.val_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.test_json_path), exist_ok=True)
        os.makedirs(self.train_image_path, exist_ok=True)
        os.makedirs(self.val_image_path, exist_ok=True)
        os.makedirs(self.test_image_path, exist_ok=True)

        # 保存数据集json文件
        with open(self.train_json_path, "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(self.val_json_path, "w") as f:
            json.dump(val_data, f, indent=4, ensure_ascii=False)
        with open(self.test_json_path, "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        # 复制图片到对应目录
        def copy_images(data_samples, target_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

            # 复制新的图片
            for sample in data_samples:
                for image_name in sample["image"]:
                    src_path = os.path.join(self.image_path, image_name)
                    dst_path = os.path.join(target_dir, image_name)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)

        copy_images(train_data, self.train_image_path)
        copy_images(val_data, self.val_image_path)
        copy_images(test_data, self.test_image_path)

        print(f"数据集划分完成:")
        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")


if __name__ == "__main__":

    # from via_sample import ViaDataSample
    # viaDataSample = ViaDataSample()

    # conversations = [
    #     Conversation("human", "<image>\n在fps游戏中..."),
    #     Conversation("gpt", "[[340, 627, 510, 892]]"),
    # ]
    # data_sample = DataSample(["aimming_no_hand.png"], conversations)
    # data_sample.save_json()

    project_path = "./scripts/dataset/input/fps_demo"
    via = ViaDataSample(project_path)
    data_processor = DatasetProcessor(via)
    data_processor.process()
