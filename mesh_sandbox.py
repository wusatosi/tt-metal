import ttnn
import torch
from collections import defaultdict

# # Open our 1x2 MeshDevice
# mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(3, 2))

# # Initialize a torch tensor
# torch_tensor = torch.zeros(6, 1, 32, 64)
# torch_tensor[..., 0:32] = 1.0
# torch_tensor[..., 32:64] = 2.0

# for x in range(0, 6):
#     torch_tensor[x,0,0,0] = x

# # Convert to ttnn.Tensor; MeshTensor holds buffers to two shards in host-memory
# mesh_tensor = ttnn.from_torch(
#     torch_tensor,
#     device=mesh_device,
#     mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
#     layout=ttnn.TILE_LAYOUT,
# )

# print(mesh_tensor)

# ttnn.visualize_mesh_device(mesh_device, tensor=mesh_tensor)

# # ===========

# mesh_tensor = ttnn.from_torch(
#     torch_tensor,
#     device=mesh_device,
#     mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (mesh_device.shape.num_rows, mesh_device.shape.num_cols), (None,3)),
#     layout=ttnn.TILE_LAYOUT,
# )
# print(mesh_tensor)
# ttnn.visualize_mesh_device(mesh_device, tensor=mesh_tensor)

# # ===========

# submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, 2), ttnn.MeshType.Ring)
# print(submeshes)

# class TestModel:
#     def __init__():
#         pass

#     def forward(self, activations, weight, bias):
#         return ttnn.linear(activations, weight, bias=bias)


# mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
# mesh_device.reshape(ttnn.MeshShape(4, 2))

# torch_tensor = torch.zeros(8, 1, 512, 512)
# for x in range(0, 8):
#     torch_tensor[x,0,0,0] = x

# torch_weight = torch.ones(512, 32)
# torch_bias = torch.ones(512, 32)


# mesh_activation = ttnn.from_torch(
#     torch_tensor,
#     device=mesh_device,
#     mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
#     layout=ttnn.TILE_LAYOUT,
# )

# mesh_weight = ttnn.from_torch(
#     torch_weight,
#     device = mesh_device,
#     mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device),
#     layout = ttnn.TILE_LAYOUT
# )

# mesh_bias = ttnn.from_torch(
#     torch_bias,
#     device = mesh_device,
#     mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device),
#     layout = ttnn.TILE_LAYOUT
# )

# mesh_output = ttnn.linear(mesh_activation, mesh_weight, bias=mesh_bias)

# # print(mesh_output)
# ttnn.visualize_mesh_device(mesh_device, tensor=mesh_activation)
# ttnn.visualize_mesh_device(mesh_device, tensor=mesh_output)

# with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
#     mesh_host_out = ttnn.to_torch(mesh_output)

# output_tensor = ttnn.all_gather(mesh_output, dim=0, cluster_axis=0, mesh_device=mesh_device)
# output_tensor = ttnn.all_gather(mesh_output, dim=0, num_links=1, topology=ttnn.Topology.Linear)

# print(mesh_host_out.shape)

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
# mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(8, 1))
mesh_device.reshape(ttnn.MeshShape(8, 1))  # row=8, cols=1

print(f"{mesh_device.shape}")
# torch_weight = torch.zeros(1, 1, 32, 4096)
torch_weight = torch.zeros(1, 1, 4096, 14336)

# dim = (-2, -1)
dim = (None, -1)

ttnn_tensor = ttnn.as_tensor(
    tensor=torch_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dim, mesh_shape=list(mesh_device.shape)),
)

# ttnn_tensor = ttnn.from_torch(
#     tensor=torch_weight,
#     dtype=ttnn.bfloat8_b,
#     layout=ttnn.TILE_LAYOUT,
#     device=mesh_device,
#     memory_config=ttnn.DRAM_MEMORY_CONFIG,
#     mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dim, mesh_shape=list(mesh_device.shape))
# )

# print("ttnn.synchronize_devices")
ttnn.synchronize_devices(mesh_device)
# print(ttnn_tensor)
# print(ttnn_tensor.shape)

ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)

# print("ttnn.ttnn.close_mesh_device")
ttnn.close_mesh_device(mesh_device)

###//// 01/18
