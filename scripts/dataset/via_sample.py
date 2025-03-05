import json
import os


def raw2normalized(box, h, w):
    box = [
        int(box[0] / w * 1000),
        int(box[1] / h * 1000),
        int(box[2] / w * 1000),
        int(box[3] / h * 1000),
    ]
    box = [min(b, 1000) for b in box]
    return box


class ViaDataSample(object):
    def __init__(self, project_path: str = "./scripts/dataset/input/fps_demo"):
        self.project_path = project_path
        self.csv_path = os.path.join(project_path, "via.csv")

    def get_bug_description(self, data_sample):
        try:
            file_attrs = json.loads(data_sample["file_attributes"])
            return file_attrs.get("bugDesc", "")
        except (json.JSONDecodeError, KeyError):
            return ""

    def get_bonding_box(self, data_sample, normalized=True):
        """
        normalized: if True, return normalized box coordinates(归一化坐标), else return raw box coordinates(像素坐标)
        """
        try:
            file_attrs = json.loads(data_sample["region_shape_attributes"])
            box = [
                file_attrs.get("x", 0),
                file_attrs.get("y", 0),
                file_attrs.get("x", 0) + file_attrs.get("width", 0),
                file_attrs.get("y", 0) + file_attrs.get("height", 0),
            ]
            if normalized:
                box = raw2normalized(
                    box, file_attrs.get("height", 0), file_attrs.get("width", 0)
                )

            return box

        except (json.JSONDecodeError, KeyError):
            return ""

    def get_bug_type(self, data_sample):
        try:
            file_attrs = json.loads(data_sample["region_attributes"])
            return file_attrs.get("bugType", "")
        except (json.JSONDecodeError, KeyError):
            return ""
