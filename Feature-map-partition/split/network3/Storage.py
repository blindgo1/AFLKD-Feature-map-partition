import torch

class StorageNode:
    def __init__(self):
        self.storage = {}

    def store_data(self, key: str, value: torch.Tensor):
        """存储数据到字典中"""
        self.storage[key] = value

    def retrieve_data(self, key: str) -> torch.Tensor:
        """通过 key 获取数据"""
        return self.storage.get(key, None)

    def delete_data(self, key: str):
        """删除指定 key 的数据"""
        if key in self.storage:
            del self.storage[key]

    def print_storage(self):
        for key, value in self.storage.items():
            print(f"Key: {key}, Value: {value}")