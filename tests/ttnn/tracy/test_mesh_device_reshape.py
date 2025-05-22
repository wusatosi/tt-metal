import ttnn
import os

# Enable profiler
os.environ["TT_METAL_CLEAR_L1"] = "1"
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

# Open a MeshDevice with the shape (1, 8)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))

print(mesh_device.shape)
# Reshape the mesh configuration to (2, 4)
mesh_device.reshape(ttnn.MeshShape(2, 4))

ttnn.DumpDeviceProfiler(mesh_device)

print(mesh_device.shape)
# Close the device; this action dumps the profiler results.
# An exception will occur here:
#   `Coordinate MeshCoordinate([1, 0]) is out of bounds for shape MeshShape([1, 8])`
ttnn.close_mesh_device(mesh_device)
