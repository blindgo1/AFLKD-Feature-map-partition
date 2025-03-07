import os

import torch
from Storage import StorageNode
import torch.distributed.rpc as rpc
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['TP_SOCKET_IFNAME'] = 'eth0'
global storage
storage = StorageNode()


def save_padding(key,tensor):
    storage.store_data(key, tensor)

def retrieve_padding(key):
    padding = storage.retrieve_data(key)
    return padding

def print_all():
    storage.print_storage()



#训练函数,将最后卷积的结果取出来并拼接再进行最后的全连接操作
def fc_training(model_part3_rref, input_tensor,rank):
    storage.store_data(f'fc_{rank}', input_tensor)
    # 当两个节点都将最后的卷积结果传输到storage中时
    while ((storage.retrieve_data(f'fc_2') is None) or (storage.retrieve_data(f'fc_3') is None)):
        pass
    # 拼接最后结果
    tensor = torch.cat(
        (storage.retrieve_data(f'fc_2'), storage.retrieve_data(f'fc_3')), 2)
    print('fc:',tensor.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_part3 = model_part3_rref.local_value().to(device)
    output = model_part3(tensor.to(device))
    return output


def init_rpc(rank, world_size):
    # 初始化rpc
    print('begin init rpc')
    options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://192.168.3.24:29500",
        _transports=["uv"]
    )
    # 当需要cuda间的传递时需要设置device_map
    # options.set_device_map('worker1', {0: 0})
    # 初始化 RPC，设置 rank 和 world_size
    rpc.init_rpc("raspberrypi8", rank=rank, world_size=world_size, rpc_backend_options=options)

    print('end init rpc')




if __name__ == "__main__":
    init_rpc(4,5)
    #仅进行初始化并等待调用
    # 关闭 RPC
    rpc.shutdown()