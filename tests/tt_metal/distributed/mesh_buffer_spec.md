# TT-Mesh Multi-Device Sharding API Spec, Next Steps and Testing Plan

## Limitations
Since an ND Tensor is flattened to 2D along the y-axis when its written to a MeshBuffer, generic 3D/4D sharding is not currently fully supported. 

Given a generic ND tensor, it can be sharded along the batch-dim (outermost dim), x-dim (innermost dim). Sharding along the y-dim is currently only supported when the batch and z-dims are 1 (simple 2D tensor case). Sharding along the z-dim is currently only supported when the batch-dim is 1 (simple 3d tensor case).

If this infra is to be exposed to TTNN in its existing state, model writers would be need to transpose their tensors before sharding along the y-dim or z-dim.

## Sharded Buffer Config
The struct below can be used to Shard + Replicate data across devices in a Mesh, through a MeshBuffer. 
```cpp
struct  ShardedBufferConfig  {
	// The MeshDevice on which the associated buffer will be allocated
	MeshDevice*  mesh_device;
	// Global buffer size. This is the size of the data that will get written to this buffer.
	// Each device will get a fraction of this size, if we purely shard (no replication).
	DeviceAddr  global_buffer_size;
	// The shape of each shard sent to each device - used to determine the data-distribution scheme (x, y).
	// 0 indicates that data along the associated dimension will be replicated.
	// If nothing is specified, default to 2D replication.
	std::pair<size_t, size_t> shard_shape = {0, 0};
	// Global shape of the buffer (x, y). Each device will get a fraction of this shape if we purely shard.
	// Given a 4D tensor, this shape is derived as <x, batch * z * y>.
	// The global_buffer_size can be derived as std::get<0>(global_buffer_shape) * std::get<1>(global_buffer_shape) * datum_size
	std::pair<size_t,  size_t>  global_buffer_shape = {0,  0};
	// Determines whether shards are written to devices across a Mesh in Row or Col major order
	ShardOrientation  mesh_shard_orientation = ShardOrientation::ROW_MAJOR;
	// Specifies how each replicated shard is laid out across Memory Banks on a single device. See spec for more details.
	DeviceLocalLayoutConfig  device_shard_layout;
};
```
The  TT-Mesh sharding infra always reads shards in Row Major order. `mesh_shard_orientation= ShardOrientation::ROW_MAJOR` is used when shards are to be placed along a mesh row in the same order as they are read (sharding across columns). Conversely,  `mesh_shard_orientation= ShardOrientation::COL_MAJOR`  is used when shards are to be placed along a mesh column in the order that they are read (sharding across rows). If replication is to be done, it will be done along the dimension opposite to the `mesh_shard_orientation`.

### Lowering TTNN Mesh-Mappers to the Sharded Buffer Config
TTNN currently supports the following `ShardTensorToMesh(mesh_device, dim)`and `ShardTensor2DMesh(mesh_device, mesh_shape, [dim_0, dim_1])`.

Below we provide examples displaying how the Mesh-Mappers distribute data on a T3K (2 rows, 4 columns), and how identical data distribution can be achieved through the Sharded Buffer Config.
#### Example 1: Height Sharding Across Columns with Width Replication Across Rows
- **Tensor Shape**: `[b = 4, z = 3, y = 32, x = 32]`
- **Mesh Mapper**: `ShardTensor2DMesh(t3k_mesh_device, dims=[None, 0])`
- **Description**: Shard the tensor along the b-dimension across mesh-columns. Replicate the shards along the mesh-rows. Since we have have 4 columns in the mesh, and we shard across the b=4 dimension, each device gets a tensor of shape `[z = 3, y = 32, x = 32]`. 

**Equivalent Sharded Buffer Config**:
- The global buffer shape in this case is `(x = 32, y = 4 * 3 * 32)`. This represents a 4D tensor being flattened to a 2D tensor acros the y axis.
- The shard shape is `(x = 0, y = 96)`. This indicates that the x dimension will be replicated and the y dimension will be sharded, with shard height 96. This corresponds to Height Sharding and Width Replication.
- The direction along which the shards are written is specified in the `mesh_shard_orientation` field. Since the Mesh-Mapper writes individual shards across columns (along mesh rows) and replicates them across rows (along mesh columns), the shard orientation is Row Major. For 1D configs (height or width), sharding will always occur along the direction specified in the shard orientation. If one of the dimensions is to be replicated, the infra will do so along the opposite direction, i.e. in this case, shards of size `(x = 32, y = 96)` will be replicated across rows (or along columns: Column Major Replication).

The final config is as follows:

```cpp
ShardedBufferConfig  {
	t3k_mesh_device,
	4 * 3 * 32 * 32 * datum_size, // global buffer size: datum size specified by user
	{0, 96}, // shard shape derived above
	(32, 4 * 3 * 32), // global buffer shape
	ShardOrientation::ROW_MAJOR, // Write shards across columns. x = 0 in the shard shape indicates that for each column, a shard of shape (x = 32, y = 96) is replicated across the 'minor' dim (row dim/vertical dim).
	device_shard_layout // As specified by user: configures how each shard is laid out inside each chip
}
```
#### Example 2: Width Sharding Across Rows with Replication Across Columns

- **Tensor Shape**: `[b = 32, z = 3, y = 128, x = 256]`
- **Mesh Mapper**: `ShardTensor2DMesh(t3k_mesh_device, dims = [3, None])`
- **Description**: Shard the tensor along the x (innermost) dimension across mesh-rows (along mesh-columns). Replicate each shard across the mesh-columns. Since we have 2 rows in the mesh, and we shard across the x=256 dimension, each device gets a tensor of shape `[b = 32, z = 3, y = 128, x = 128]`.

**Equivalent Sharded Buffer Config**:
- The global buffer shape is `(x = 256, y = 32 * 3 * 128)`. This represents a 4D tensor being flattened to a 2D tensor along the y dimension.
- The shard shape is `(x = 128, y = 0)` . This indicates that the y dimension will be replicated and the x dimension will be sharded, with a shard width of 128. This corresponds to Width Sharding and Height Replication.
- Since the Mesh-Mapper writes individual shards across rows (along mesh columns) and replicates them across columns (along mesh rows), the shard orientation is Col Major. In this case, each device gets a shard of size `(x = 128, y = 32 * 3 * 128)`

The final config is as follows:
```cpp
ShardedBufferConfig  {
	t3k_mesh_device,
	32 * 3 * 128 * 256 * datum_size, // global buffer size: datum size specified by user
	{128, 0}, // shard shape derived above
	(256, 32 * 3 * 128), // global buffer shape
	ShardOrientation::COL_MAJOR, // Write shards across rows. y = 0 in the shard shape indicates that for each row, a shard of shape (x = 128, y = 32 * 3 * 128) is replicated across the 'minor' dim (col dim/horizontal dim).
	device_shard_layout // As specified by user: configures how each shard is laid out inside each chip
}
```

#### Example 3: Row Major Block Sharding
**Tensor Shape**: `[b = 1, z = 1, y = 128, x = 256]`
**Mesh Mapper**: `ShardTensor2DMesh(t3k_mesh_device, dims = [2, 3])`
**Description**: Shard the tensor along the x and y dimension across mesh-columns and mesh-rows, respectively. Since we have 2 rows and 4 columns in the mesh, each device gets a shard of shape `b = 1, z = 1, y = 64, x = 64`.

**Equivalent Sharded Buffer Config**:
- The global buffer shape is `(x = 256, y = 128)`. This represents a 4D tensor being flattened to a 2D tensor along the y dimension.
- The shard shape is `x = 64, y = 64)`. This indicates that the configuration corresponds to Block Sharding.
- Since the Mesh Mapper shards the x dimension across columns and the y dimension across rows, the shard orientation is Row Major. 

The final config is as follows:
```cpp
ShardedBufferConfig  {
	t3k_mesh_device,
	1 * 1 * 128 * 256 * datum_size, // global buffer size: datum size specified by user
	{64, 64}, // shard shape derived above
	(256, 128), // global buffer shape
	ShardOrientation::ROW_MAJOR, // Write shards across rows, wrap around then write to the next column.
	device_shard_layout // As specified by user: configures how each shard is laid out inside each chip
}
```
## Next Steps
Given a 4D Tensor and a Mesh-Mapper, add a set of APIs to generate the Sharded Mesh Buffer Config (shard orientation, shard shape, global buffer shape and global buffer size). The goal is to lower the Torch/xTensor representation of multi-device data-distribution to the format presented above, which TT-Mesh understands. TT-Mesh provides a more generic and complex interface to perform data-distribution, since it operates with buffers that can have more arbitrary data layouts.

This layer serves as the bridge between TTNN Tensors and TT-Mesh Buffers, replacing the torch based multi-device sharding functionality used in TTNN.

The "lowering" APIs should assert if the data distribution strategy for the tensor is not supported by the existing sharding infra. Limitations are listed in the first section of this doc (ND sharding is not supported). The goal is eventually to remove this limitation.

## Testing
- Sweeps/Randomized Testing (valid cases ... no ND sharding): 
	- Generate several valid sharding + replication configs to sweep over, each with a random tensor
	- Generate golden results for each config in C++ using xTensor, in the form of individual tensor shards. Flatten each tensor to a 2D Tensor (equivalent to a MeshBuffer). 
	- Lower each config to an equivalent `ShardedBufferConfig` . Write the tensor data to device using this config (please see `TestSharding` inside `test_mesh_workload.cpp`), read it back and verify that it matches the shards generated by xTensor.
- Adversarial Testing (invalid cases requiring ND sharding):
	- Generarte several invalid sharding + replication configs and ensure that an assert is hit when trying to lower these to the `ShardedBufferConfig`
- Parity Testing:
		- Get a list of all data distribution strategies used by models relying on TTNN. Generate random tensors for each config
		- Verify that for all cases the data that is written to device using the equivalent `ShardedBufferConfig` is valid by reading back individual shards. Tests should either pass or hit an assert if the strategy is not supported.