import os
import yaml


class Config:
    def __init__(self, filename, encoding="utf-8"):
        self.filename = filename
        self.encoding = encoding
        self.suffix = self.filename.split(".")[-1]
        if self.suffix not in ["yaml", "yml"]:
            raise ValueError("不能识别的配置文件后缀：{}".format(self.suffix))

    def parse(self):
        """
        解析yaml
        :return:
        """
        with open(self.filename, "r", encoding=self.encoding) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data


if __name__ == "__main__":
    current_dir = os.getcwd()
    cm = Config(current_dir + "/application.yml")
    res = cm.parse()
    print(res)
