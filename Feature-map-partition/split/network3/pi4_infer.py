import os

import torch
import torch.distributed.rpc as rpc
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['TP_SOCKET_IFNAME'] = 'eth0'

def training4(model_part2_rref, input_tensor,rank):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_part2_rref.to_here()
    output = model(input_tensor.to(device),rank)
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
    rpc.init_rpc("raspberrypi4", rank=rank, world_size=world_size, rpc_backend_options=options)

    print('end init rpc')

if __name__ == "__main__":
    init_rpc(2,7)
    #仅进行初始化并等待调用
    # 关闭 RPC
    rpc.shutdown()