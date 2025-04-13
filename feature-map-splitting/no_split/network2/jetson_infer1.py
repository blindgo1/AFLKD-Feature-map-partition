import os
import time
import torch.distributed.rpc as rpc

import torchvision.transforms as transforms
from pi4_infer import training4
from pi6_infer import training6
from model import VGG16Small_part1,create_model_part2_instance,create_model_part3_instance


os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['TP_SOCKET_IFNAME'] = 'eth0'

import torch
from PIL import Image

def jetson1_inference():
    #推理图片
    filename = ("../../input/dog.jpg")
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    images = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('data_preprocess finish')
    # 定义网络模型 part1  统一都定义成rref格式
    model_part1_rref = rpc.remote(f"jetson1", VGG16Small_part1)
    # 由于需要用rpc来进行模型并行，因此需要定义一个远程rref对象来给后续节点调用
    model_part2_rref = rpc.remote(f"raspberrypi4", create_model_part2_instance)
    model_part3_rref = rpc.remote(f"raspberrypi6", create_model_part3_instance)

    model_part1 = model_part1_rref.local_value().to(device)
    #开始推理时间
    start_time = time.time()
    intermediate_output = model_part1(input_tensor.to(device))  # 将输入数据传递给模型part1
    print(intermediate_output.shape)  # 打印中间结果的形状
    # 将中间结果发送到raspberry上进行调用
    intermediate_output = intermediate_output.to('cpu')  # 将中间结果移动到cpu
    intermediate_output2 = rpc.rpc_sync(f"raspberrypi4", training4,
                          args=(model_part2_rref, intermediate_output,))  # 将中间结果发送到raspberry上进行调用
    print(intermediate_output2.shape)
    output = rpc.rpc_sync(f"raspberrypi6", training6,
                          args=(model_part3_rref, intermediate_output2,))
    print(output.shape)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.6f} seconds")
    rpc.shutdown()


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
    rpc.init_rpc("jetson1", rank=rank, world_size=world_size, rpc_backend_options=options)
    print('end init rpc')


if __name__ == "__main__":
    init_rpc(0, 3)
    jetson1_inference()
