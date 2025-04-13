import os
import time
import torch.distributed.rpc as rpc
from store_node import save_padding,fc_training,retrieve_padding
from pi6_infer import training6
from pi8_infer import training8
import torchvision.transforms as transforms

from model import VGG16Small_part1,create_model_part2_instance,create_model_part3_instance,create_model_part4_instance

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['TP_SOCKET_IFNAME'] = 'eth0'
import torch
from PIL import Image

# 全局变量来存储接收到的数据集
# global received_dataset
# received_dataset = None
# def receive_data(dataset):
#     received_dataset = dataset
#     print("Data received on jetson1")

def jetson1_inference(rank ):
    filename = ("/home/wxg/Documents/fjq/input/dog.jpg")
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_tensor = preprocess(input_image)
    images = input_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('data_preprocess finish')

    #定义网络模型 part1  统一都定义成rref格式
    model_part1_rref = rpc.remote(f"jetson2", VGG16Small_part1)
    #由于需要用rpc来进行模型并行，因此需要定义一个远程rref对象来给后续节点调用
    model_part2_rref = rpc.remote(f"raspberrypi6", create_model_part2_instance)
    model_part3_rref = rpc.remote(f"raspberrypi10", create_model_part3_instance)
    model_part4_rref = rpc.remote(f"raspberrypi8", create_model_part4_instance)
    model_part1 = model_part1_rref.local_value().to(device)

    # 开始推理

    with torch.no_grad():
        # for images, labels in train_loader:
            #现将输入数据按照高度维度进行分割，分成两份，分别存储在images[0]和images[1]
            print('input image size:', images.shape)
            images = torch.chunk(images, 2, dim=2)
            local_input = images[rank]
            #对初始分割数据进行填充
            if rank == 0:
                #将最开始的填充数据存入storage中
                padding = images[rank + 1][:, :, 0, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv0_1', padding.to('cpu')))
            if rank == 1:
                padding = images[rank - 1][:, :, -1, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv0_0', padding.to('cpu')))

            local_input = local_input.to(device)
            start_time = time.time()
            # 将输入数据传递给模型part1
            intermediate_output = model_part1(local_input, rank)
            #将maxpool后的padding存入存储节点
            if rank == 0:
                padding = intermediate_output[:, :, -1, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv3_2', padding.to('cpu')))
            if rank == 1:
                padding = intermediate_output[:, :, 0, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv3_3', padding.to('cpu')))
                #将中间结果发送到raspberry上进行调用
            intermediate_output = intermediate_output.to('cpu') # 将中间结果移动到cpu
            intermediate_output2 = rpc.rpc_sync(f"raspberrypi6", training6,args=(model_part2_rref, intermediate_output, 3)) # 将中间结果发送到raspberry上进行调用
            if rank == 0:
                padding = intermediate_output2[:, :, -1, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv5_4', padding.to('cpu')))
            if rank == 1:
                padding = intermediate_output2[:, :, 0, :].unsqueeze(2)
                rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv5_5', padding.to('cpu')))

            intermediate_output3 = rpc.rpc_sync(f"raspberrypi10", training8,
                                                args=(model_part3_rref, intermediate_output2, 5))
            output = rpc.rpc_sync(f"raspberrypi8", fc_training, args=(model_part4_rref, intermediate_output3, 5,))
            print('output:',output.shape)
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.6f} seconds")
    rpc.shutdown()


def init_rpc(rank,world_size):
    # 初始化rpc
    print('begin init rpc')
    options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://192.168.3.24:29500",
        _transports=["uv"]
    )
    #当需要cuda间的传递时需要设置device_map
    #options.set_device_map('worker1', {0: 0})
    # 初始化 RPC，设置 rank 和 world_size
    rpc.init_rpc("jetson2", rank=rank, world_size=world_size, rpc_backend_options=options)
    print('end init rpc')
if __name__ == "__main__":
    init_rpc(1,7)
    jetson1_inference(1)
