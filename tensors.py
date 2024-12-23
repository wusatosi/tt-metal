import ttnn
import torch


device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_a = torch.randn((64, 800), dtype=torch.bfloat16)
t1 = ttnn.from_torch(torch_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

print("t1 shape:", t1.shape)
print("t1 dtype:", t1.dtype)
print("t1 layout:", t1.layout)
print(t1)
mem_config = ttnn.create_sharded_memory_config(
    shape=t1.shape, core_grid=ttnn.CoreGrid(x=2, y=2), strategy=ttnn.ShardStrategy.WIDTH
)
t1 = ttnn.to_memory_config(t1, mem_config)

print("t1 shape:", t1.shape)
print("t1 dtype:", t1.dtype)
print("t1 layout:", t1.layout)
print(t1)

ttnn.close_device(device)
