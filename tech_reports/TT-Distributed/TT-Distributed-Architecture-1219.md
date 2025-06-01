# TT-Metalium Distributed

Authors: TT-Metalium Scale-Out Team

For questions and comments please use the [TT-Metalium Scale-Out Discord Server](https://discord.com/channels/863154240319258674/1321621251269328956)

## Architecture Specification

Version 1.0

[1. Overview](#overview)

[2. Background and Project Motivation](#background)
  - [2.1 Virtualization through TTNN](#virtualization-through-ttnn)
  - [2.2 Project Motivation and Design](#motivation)
  - [2.3 Dependencies with External Efforts](#dependencies)

[3. TT-Mesh](#tt-mesh)
  - [3.1 MeshDevice: Overview and Associated Data-Structures](#meshdevice)
    - [3.1.1 Terminology](#meshdevice-terminology)
    - [3.1.2 Constraints and Properties of a Virtual Mesh](#meshdevice-constraints)
    - [3.1.3 MeshDevice Abstraction](#meshdevice-abstraction)
    - [3.1.4 Data Structures](#meshdevice-data-structures)
    - [3.1.5 Lightweight and Consistent APIs](#meshdevice-lightweight-and-consistent-apis)
  - [3.2 Virtual Command Queues](#virtual-command-queues)
  - [3.3 Memory Management: MeshBuffer and MeshAllocator](#meshbuffer)
  - [3.4 MeshWorkload: Overview, Data-Structures and APIs](#meshworkload)
  - [3.5 MeshEvent: Data-Structure and APIs](#meshevent)
  - [3.6 MeshTrace: Overview and APIs](#meshtrace)
  - [3.7 End to End Programming Example](#tt-mesh-end-to-end)
  - [3.8 MeshCommandQueue: Data Movement to and from a TT-Mesh](#meshcommandqueue)
  - [3.9 Host Runtime Design: Conservative Multithreading](#mesh-host-runtime-design)
  - [3.10 MeshWorkload: Implementation Details](#meshworkload-details)
  - [3.11 MeshEvents: Implementation Details](#meshevent-details)
  - [3.12 MeshTrace Implementation Details](#meshtrace-details)
  - [3.13 Summary: Dependencies, APIs and Data-Structures on Host](#tt-mesh-summary)

[4. TT-Distributed](#tt-distributed)
  - [4.1 Offline System Descriptor: Caching UMD Queries](#offline-system-descriptor)
  - [4.2 DistributedDevice](#distributed-device)
  - [4.3 DistributedBuffer](#distributed-buffer)
  - [4.4 DistributedWorkload: Overview, Data-Structures and APIs](#distributed-workload)
  - [4.5 DistributedEvents: Overview, Data-Structures and APIs](#distributed-events)
  - [4.6 DistributedTrace: Overview and APIs](#distributed-trace)
  - [4.7 Command Serialization](#command-serialization)
  - [4.8 Network Transport](#network-transport)

[5. TT-NN Integration](#tt-nn-integration)
  - [5.1 TT-Mesh/TT-Distributed Interoperability Layer](#tt-mesh-tt-distributed-interoperability)

[Appendix](#appendix)
 - [Appendix A: Existing Runtime Architecture Overview](#appendix-a)
    - [A.1 Current State of Metal Runtime](#appendix-a-current-state-of-runtime)
    - [A.2 A More Modular Approach to Metal Runtime](#appendix-a-more-modular-approach)
    - [A.4 Distributed Metal Runtime Using Existing Components and Virtualization](#appendix-a-distributed-metal-runtime-using-existing-components-and-virtualization)
 - [Appendix B: UMD](#appendix-b)
     - [Appendix B.1: UMD Queries](#appendix-b-umd-queries)

# Overview <a id="overview"></a>

This document presents a specification for **TT-Mesh** and **TT-Distributed**, Tenstorrent’s scale-up and scale-out infrastructure to natively support workloads on multiple Tenstorrent hardware accelerators, potentially spanning multiple host servers.

TT-Metalium provides a flexible programming model over a mesh of Tensix cores. The TT-Mesh and TT-Distributed layers, built on top of TT-Fabric, extend this programming model to a grid of Tensix cores spanning multiple devices connected over ethernet and controlled by multiple host processors. Through this, users are exposed to the same programming model as a single device, when systems are scaled-out.

The key idea is to present a multi-device multi-host system as: 1) a large virtual device that has, 2) a distributed shared memory space, and 3) a set of programmable Tensix cores. Unlike conventional architectures, programming cores distributed over multiple devices or multiple host machines will not fundamentally alter the programming model.

![A diagram of a computer Description automatically generated](images/image001.png)

# 2. Background and Project Motivation <a id="background"></a>

## 2.1 Virtualization through TTNN <a id="virtualization-through-ttnn"></a>

TT-NN is a library that provides a Pytorch-like interface for executing compute on Tenstorrent accelerators. This interface is available and supported for single-process, single-host environments with operations that can be dispatched synchronously and asynchronously (through a single or multithreaded runtime environment) across a mesh of devices. See [Programming Mesh of Devices](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) for more information. TT-NN builds on top of TT-Metalium to provide a high-level interface in the form of operations and tensors in a neural network op library.

The table below displays the compute paradigms supported by TTNN.

| Table 1: TT-NN Multi-Device Operation Dispatch (state as of Nov 2024) | | |
| --- | --- | --- |
|  | **Single Process** | **Multi-Process/ Multi-Host** |
| Synchronous | Supported | Unsupported |
| Asynchronous | Supported | Unsupported |

Today, TT-NN virtualizes the devices arranged in a physically connected mesh as a single MeshDevice abstraction. Users can largely be transparent to orchestration and coordination at the level of individual devices. This virtualization effectively gives users a single handle to a larger compute mesh and memory space. The compute model is exposed as a Single-Program-Multiple-Device (SPMD) environment, which broadcasts operations across all devices in a mesh.

Today, TT-Metalium does not natively support the ability to virtualize a multi-chip cluster; this abstraction is implemented entirely at the TT-NN layer. A side-effect of this is that TT-NN is responsible for lowering MeshDevice APIs into TT-Metalium primitives targeting single devices. This results in the heavy host-bound lowering process being repeated across devices. If done serially, this almost always results in multi-chip workloads being bottlenecked on host.

As an optimization, TT-NN uses multi-threading to implement asynchronous host dispatch across devices, with a single dispatch thread assigned to each device. An example of the SPMD compute model and asynchronous TT-NN dispatch is provided in the diagram below, displaying how a simple unary operation is lowered to a Mesh of eight devices.

![](images/image002.png)

A direct consequence of the existing multi-threaded dispatch infrastructure is an overloaded host. The impacts of this problem are not visible for workloads running on small clusters but this does not scale. The number of threads the host needs to manage is directly proportional to the number of devices in the mesh.

As shown in the diagram below, for a system with 64 devices (a TGG Cluster), host runtime spawns 192 threads (the exact assignment of threads is described in a later section). Even on a processor with 96 cores, with threads optimally bound, each CPU core is running at least 2 threads and is significantly overloaded. The effects of this behavior are clearly visible when running simple models on large cluster using the existing approach.

![](images/image006.png)

This problem will get worse for larger clusters. Thus, the existing virtualization infrastructure is not a viable solution for scale-out.

## 2.2 Project Motivation and Design <a id="motivation"></a>

The diagram above presents a motivating case for moving the virtualization lower in the software stack. Virtualization at the TT-NN layer and multi-threaded dispatch does not scale and is unable to virtualize across different servers. The motivation and scope of the problem to solve are the following:

1. **Extend the TT-Metalium programming model to natively support multi-device and multi-host**. This is decoupled as an “Op Design Problem” and a “Dispatch Problem”. Op/Kernel writers using the TT-Distributed infrastructure have the flexibility to target a larger compute mesh while being entirely transparent to the physical resources used for dispatching the workload to TT-Accelerators across servers.
   1. TT-Distributed is agnostic of single-host / multi-host configurations

![](images/image007.png)

Figure (A): Separation of the Op Design and Dispatch Problems

* 1. Figure (B) displays TT-Distributed exposing two different physical configurations as a **single Virtualized Mesh of TT-Accelerators**

![](images/image008.png)

Figure (B): TT-Distributed exposes two 4x2 and a single 4x4 Cluster as the same Virtual Mesh

1. Solve scalability issues seen with the TT-NN virtualization layer.
2. **Introduce Unified Distributed Memory across the entire Virtual Mesh** (including Multi-host systems)
   1. Introduce MeshBuffer – a buffer that is allocated across a Virtual Mesh and can be allocated in lock-step across physical devices
   2. MeshBuffer across a mesh of devices is an extension of Buffer across a mesh of cores
   3. Any core on any each physical device within the Virtual Mesh can Read/Write to any other address in the Mesh (Unified Distributed Memory)

This architecture specification addresses these points through the proposed **TT-Mesh** and **TT-Distributed** frameworks**.**

Built on top of TT-Fabric, and existing TT-Metalium runtime infrastructure, TT-Mesh integrates the MeshDevice concept natively with TT-Metalium while TT-Distributed unlocks multi-host support. Through this, we introduce:

* A unified programming model between a single device and a mesh of devices, as users interact with more Tensix cores and larger distributed memory regions across multiple servers. This transparently extends the existing TT-Metalium programming model from a “Mesh of Cores” to a “Mesh of Cores contained in a Mesh of Devices across servers”.
* The ability to use TT-Fabric (Tenstorrent’s multichip routing firmware) for parallelizing dispatch and data-movement across the mesh, using chip-to-chip broadcasts. This allows the physical resources supporting parallelism to scale with the size of the compute cluster, removing bottlenecks on Host that exist with the existing infrastructure.
* A unified memory model exposed to users as they scale across multiple devices connected to different servers.

The diagram below displays the high-level architecture of the proposed runtime stack, reusing concepts from the single-device runtime domain, and built on top of existing layers.

The **TT-Mesh** infrastructure introduces MeshDevice, MeshBuffer, MeshWorkload, MeshTrace and MeshEvents objects; based on single-device analogs.

The **TT-Distributed** infrastructure introduces DistributedDevice, DistributedBuffer, DistributedWorkload, DistributedTrace, DistributedEvent variants of these objects, which can exist across multiple hosts, through serialization and transport layers.

![](images/image009.png)

## 2.3 Dependencies with External Efforts <a id="dependencies"></a>

This effort is heavily entangled with several on-going and scheduled efforts on the Runtime Infrastructure Roadmap.

A list of the dependencies, and components they block is provided below. A detailed description of each component will be provided in the sections below.

| **Project Dependency** | **Component(s) being Blocked** |
| --- | --- |
| Coordinate Virtualization | MeshWorkload Implementation (**V1**)  Ability to build kernel binaries offline on a controller host, not directly connected to a Physical Device (**V1.2**) |
| Finalized Host API Contract using Handles | Finalized Distributed and Mesh APIs (**V1)** |
| Separating Build from Device | Finalized MeshDevice and DistributedDevice Implementation using a JIT Build System (**V1.2**)  Ability to build kernel binaries offline on a controller host, not directly connected to a Physical Device (**V1.2**) |
| Modular Dispatch Management Infrastructure | Finalized MeshDevice and DistributedDevice Implementation using a dedicated Control Plane.  MeshDevice Dispatch integration with TT-Fabric (**V1.2**) |
| TT-Fabric: Generic Routing over Ethernet | Broadcast based Dispatch (**V1.2**) |

# 4. TT-Distributed <a id="tt-distributed"></a>

TT-Distributed is the proposed solution for interfacing with a mesh of accelerators spanning multiple host systems.

![](images/image038.png)

## 4.1 Offline System Descriptor: Caching UMD Queries <a id="offline-system-descriptor"></a>

TT-Metal exposes several APIs to query physical device state through the tt\_cluster interface during runtime. Additionally, setting up the local or distributed session requires the ability to query system parameters, not directly exposed to users, through tt\_cluster.

Some of the APIs exposed by these layers query device attributes through UMD, over PCIe/Ethernet. Others rely on querying predefined configuration files on disk, through tt\_cluster. A list of these APIs is provided below and categorized according to the data-paths used.

Query APIs using the UMD layer need to be accessible on the Controller Host, which may not be physically connected to the Virtual Mesh it manages. This information will be propagated to (and potentially cached on) the Controller as part of the initial Distributed Runtime setup/handshaking process across hosts.

During this process, a set of Remote Hosts, physically connected to Meshes are requested to query the required parameters through local UMD sessions. These parameters (such as harvesting masks, the worker grid size, Mesh Topology/number of devices) span the entire Mesh and are embedded inside a single or multiple configuration files and sent back to the Controller over the network.

At minimum, the Controller needs to have the following information (Remote Executors need to query and forward physical device parameters that allow the Controller to construct this state):

* UMD Derived SOC Descriptors + Harvesting for each chip in the Physical Mesh
* Core Descriptors (Contains Dispatch Core Assignment Information)
* Accelerator Mesh Topology and connectivity to Remote Hosts (Cluster Descriptor)
* Aggregated Physical Mesh Topology, across all Remote Hosts (Mesh Descriptor)
* Miscellaneous Physical Parameters: Architecture + Product type, PCIe/Ethernet Link state, Fmax, etc.

Using the configuration files forwarded over the RPC stack, or loaded from a file system cache, the controller creates a Virtual Query Layer, through its Distributed Session Setup Manager.

The query layer consists of a Virtual Cluster Interface, built up from the parameters and objects listed above. This interface is identical to the existing tt\_cluster interface (to ensure that the distributed and local runtime environments have the same user experience) but is constructed offline, by parsing configuration files. The constructor for the Virtual Cluster will be provided file/object handles, instead of relying on UMD.

An example of the required dataflow is presented below.

![](images/image039.png)

## 4.2 DistributedDevice <a id="distributed-device"></a>

This section describes the design for DistributedDevice, an abstraction that virtualizes over a mesh of physically connected devices that can span multiple host servers. The DistributedDevice maintains the same programming model and API contract as MeshDevice while extending capabilities across multiple hosts.

The DistributedDevice presents users with:

* A unified view of compute resources spanning multiple hosts
* Unified memory model across the entire device cluster
* Transparent workload distribution and synchronization

### 4.2.1 Terminology:

We define “Host Controller” to denote the host process responsible for managing the distributed runtime and orchestrating the dispatch of operations distributed across multiple hosts.

We define “Remote Host Executor” to denote the runtime executor responsible for servicing requests received by the Host Controller to its local physical mesh of devices.

### 4.2.2 Programming Model:

TT-Distributed extends the programming model of TT-Mesh to work in a multi-host environment. We preserve an eager-first asynchronous dispatch programming model where Remote Executors are responsible for the orchestration of data and operations dispatched onto its local accelerator mesh.

### 4.2.3 Data Structure:

```cpp
struct ControllerHostConfig {
    std::string controller_name;  // named identifier
    std::string controller_ip_address;
    uint32_t controller_id;
    uint32_t process_rank;
    uint16_t port;
};

struct RemoteHostConfig {
    std::string remote_name;  // named identifier
    std::string remote_ip_address;
    uint16_t executor_id;
    uint32_t process_rank;
    uint16_t port;

    // Device Topology
    MeshShape mesh_shape;
    Arch arch_type;
};

constexpr uint32_t MAX_REMOTE_HOSTS = 32;

struct MeshConfig {
    MeshShape shape;
    MeshOffset offset;
};

struct DistributedDeviceConfig {
    MeshConfig mesh_config;
    ControllerHostConfig host_config;
    std::array<RemoteHostConfig, MAX_REMOTE_HOSTS> remote_executor_config;
};
```

### 4.2.4 API:

```cpp
DeviceHandle CreateDistributedDevice(
    DistributedDeviceConfig config, v1::DeviceOptions opts);
```

## 4.3 DistributedBuffer <a id="distributed-buffer"></a>

This section describes the design for DistributedBuffer, a logical device buffer hierarchically composed of shards and pages of data that span multiple remote hosts interfacing with device mesh. This extends MeshBuffer containing pages which are distributed across remote host device mesh.

The same virtualization technique employed in TT-Mesh is used here. We extend the “lock-step” allocation across hosts, devices, and banks of memory.

### 4.3.1 Data Structure

```cpp
struct DistributedBuffer {
private:
    // Mesh Allocator provides an address
    BufferAddress address_;

    // Aligned size
    BufferSize size_in_bytes_;

    // Memory layout across the Virtual Mesh distributed address space
    DistributedBufferConfig distributed_config_;

public:
    // DistributedBuffer construction leads to an allocation
    static DistributedBuffer create(const DistributedBufferConfig& config);

    DeviceHandle get_device();
    DeviceAddr address();
    DeviceAddr size();
};
```

### 4.3.2 APIs

A set of creation/destruction APIs are introduced:

```cpp
// Distributed Buffer Creation and Management
BufferHandle CreateDistributedBuffer(const DistributedBufferConfig& config);

void DeallocateDistributedBuffer(BufferHandle buffer);
```

### 4.3.3 Distributed Data Loading

There are a few Distributed Data Loader strategies that can be implemented to load data onto devices spanning multiple hosts. Since the allocator is lock-step across hosts, devices and banks, we can opt to have:

1. Host Controller loads data locally and distributes to RemoteExecutor
2. Zero Copy Data Transfer: Host Controller issues a command to all RemoteExecutors to load its shard of data. DistributedShardSpec information along with process\_id\_rank uniquely identifies the shard to load from its local file store. This use-case would be applicable for training workloads and loading pretrained weights on inference.

![](images/image040.png)

The above example shows how a distributed data loader would work. An 8x8 DistributedDevice is instantiated and the controller requests a weight tensor of [256,160] to be loaded from local file store. Host Controller issues a remote request to each of the Remote Executor to load its corresponding shard of data into local host RAM and then forwarded on as a local EnqueueMeshBuffer with a MeshDevice Device Handle for interfacing with its local accelerator mesh.

## 4.4 DistributedWorkload: Overview, Data-Structures and APIs <a id="distributed-workload"></a>

This section introduces the DistributedWorkload primitive, which allows compute to be spawned on a Virtual Mesh directly through the Controller Host. This object exposes the same programming model and APIs as the MeshWorkload primitive.

We provide a summary of the associated data-structure and user-facing APIs before discussing implementation details in a later section.

The minimal functional representation of a DistributedWorkload is as follows.

```cpp
class DistributedWorkload {
private:
    // Internal container populated by user to specify workload attributes
    MeshWorkloadHandle workload_;

public:
    // ==== Modifiers identical to the MeshWorkload class ====

    // Mutate the workload_ attribute under the hood
    void add_program(const LogicalDeviceRange& device_range, const Program& program);
    void add_program(const Program& program);
    void set_runtime_args(
        const LogicalDeviceRange& device_range,
        const CoreRangeSet& core_range_set,
        KernelHandle kernel_id,
        const std::vector<uint32_t> runtime_args);

    // ============= New functionality for DistributedWorkload =============

    // ==== Not exposed directly to users and used by the Runtime layer ====

    // Check if a serialized state for this workload was generated and cached by
    // the Serialization layer
    bool is_serialized();

    // If this object is mutated post-serialization, its serialized state must
    // be invalidated. This requires communication between the Runtime and
    // Serialization layers
    void invalidate_serialized_state();

    // Get access to underlying workload object
    MeshWorkloadHandle get_workload();

    // Get access to populated serialized state for inspection
    SerializedMeshWorkloadHandle get_serialized_workload_handle();
};
```

This data-structure is built directly on top of the MeshWorkload class, through which users are exposed to identical APIs for interfacing with compute primitives. Additionally, this object contains functionality for interfacing with the Serialization Layer. This layer serves as an entry point for a DistributedWorkload to be lowered to Remote Executors, which are responsible for then constructing physical device state based on a deserialized representation of the MeshWorkload object contained in the DistributedWorkload.

More details on the lowering process are provided in the implementation section.

The APIs exposed to users for directly interfacing with a DistributedWorkload are as follows:

```cpp
// Creates an empty DistributedWorkload object containing an empty MeshWorkload
DistributedWorkload CreateDistributedWorkload();

// Wrapper around distributed_workload.add_program. By default, the added program runs
// on the entire Virtual Mesh its enqueued to.
void InsertProgramInMeshWorkload(
    DistributedWorkload& distributed_workload,
    const Program& program,
    const LogicalDeviceRange& logical_device_range = MaxDeviceRange);

// For a given Program in the contained MeshWorkload, update/mutate Runtime Args on a
// specified Device and Core Range for a kernel in the Program. The program being
// modified must exist on the logical_device_range that is specified.
void SetRuntimeArgs(
    DistributedWorkload& distributed_workload,
    Program& program,
    KernelHandle kernel,
    const LogicalDeviceRange& logical_device_range,
    const CoreRangeSet& core_ranges,
    stl::Span<const uint32_t> runtime_args);

// Dispatch a MeshWorkload to a Virtual Mesh through the Serialization, Transport,
// Deserialization and Local Device Dispatch layers
void EnqueueDistributedWorkload(
    CommandQueueHandle cq,
    std::shared_ptr<DistributedWorkload> distributed_workload,
    bool blocking);
```

## 4.5 DistributedEvents: Overview, Data-Structures and APIs <a id="distributed-events"></a>

Extending the concept of a MeshEvent to a Controller Host requires the introduction of the DistributedEvent primitive. This is the main synchronization mechanism for distributed workloads and expands the functionality of a MeshEvent. In particular, this primitive and its associated APIs can be used for (in order of increasing latency):

* Cross CQ Synchronization on the Virtual Device: No Host in Loop
* Virtual CQ to Remote Executor Synchronization: Remote Host(s) in Loop
* Virtual CQ to Controller Synchronization through Remote Executor(s): All Hosts in Loop

Fundamentally, the only difference between a MeshEvent and a DistributedEvent is the number of pipeline stages over which handshaking/synchronization is supported. Since the pipeline for distributed workloads consists of multiple hosts arranged in a hierarchical topology, the synchronization mechanisms must support Host to Mesh synchronization across all levels of the hierarchy.

All other concepts for this primitive carry over from the MeshEvent level.

The data-structure and APIs exposing this functionality are presented below.

```cpp
// Primitive object containing only synchronization information
struct DistributedEventPrimitive {
    // Internal container maintaining synchronization information at the Remote
    // Executor level
    MeshEventHandle event;

    // Additional synchronization information:
    // True if the Executor Hosts will notify the controller host of event completion
    // Only has side-effects if mesh_only is false. If mesh_only == false, Executor
    // never gets event notification and can never update Controller.
    bool propagate_to_controller_host = false;
};

// Wrapper that interfaces with the serialization layer
struct DistributedEvent {
    // Internal container maintaining end to end synchronization information
    DistributedEventPrimitiveHandle event;

    // Accessor to underlying event object
    DistributedEventPrimitiveHandle get_event();

    // Accessor to populated serialized state for inspection
    SerializedDistributedEventHandle get_serialized_event_handle();
};

// Have the specified CQ on the specified device_range record a "Mesh Local" Event.
// When this command is processed by a CQ on each physical device, an event
// notification will only be sent to other CQs on the device.
// The event update will not be propagated to any of the hosts.
void EnqueueRecordDistributedEvent(
    CommandQueueHandle command_queue_handle,
    shared_ptr<DistributedEvent> event,
    LogicalDeviceRange& device_range);

// Have the specified CQ on the device_range tied to the Event wait for a "Mesh Local" Event.
// When this command is processed by a CQ on each physical device, it will lead
// to a stall until the CQ responsible for recording the event signals completion.
void EnqueueWaitForDistributedEvent(
    shared_ptr<DistributedEvent> event,
    CommandQueueHandle command_queue_handle);

// Have the specified CQ on the Mesh record an event that propagates back to the
// dedicated Remote Executor.
// When this command is processed by a CQ on each physical device, an event
// notification will be sent to other CQs on the device and back to the Remote
// Executor through the Event Notification queue.
void EnqueueRecordDistributedEventToExecutors(
    CommandQueueHandle cq_handle,
    shared_ptr<DistributedEvent> event,
    LogicalDeviceRange& device_range);

// Have Remote Executors block until the specified event is acknowledged by the
// Virtual Mesh it is meant to be recorded on.
void DistributedEventSynchronizeOnExecutors(shared_ptr<DistributedEvent> event);

// Have the specified CQ on the Mesh record an event that propagates back to the
// Controller.
// When this command is processed by a CQ on each physical device, an event
// notification will be sent to other CQs on the device and back to the Remote
// Executor through the Event Notification queue.
// Once event completion is seen on across all physical devices, each remote executor
// will send a completion signal to the controller.
void EnqueueRecordDistributedEventToController(
    CommandQueueHandle cq_handle,
    shared_ptr<DistributedEvent> event,
    LogicalDeviceRange& device_range);

// Have the Controller block until the specified event is acknowledged by the Virtual
// Mesh it is meant to be recorded on.
// Remote Executors will process the local events and propagate completion to the
// controller, which will unblock this call
void DistributedEventSynchronizeOnController(shared_ptr<DistributedEvent> event);

// Have Controller block until the specified CQ on the mesh has completed all enqueued
// tasks.
// This calls EnqueueRecordDistributedEventToController and
// DistributedEventSynchronizeOnController under the hood.
void Finish(CommandQueueHandle command_queue_handle);
```

## 4.6 DistributedTrace: Overview and APIs <a id="distributed-trace"></a>

The DistributedTrace framework exposes the ability to trace workloads running across multiple hosts. The metadata associated with the traced workload will exist in memory distributed across the entire Virtual Mesh mapped to the controller, through the DistributedBuffer class.

The APIs for interfacing with a DistributedTrace object are nearly identical to their MeshTrace counterparts and are presented below.

```cpp
// Start capturing a DistributedTrace. Returns a handle to the DistributedTrace
// object currently being captured. Any Fast Dispatch commands on the
// specified cq_id between this call and EndDistributedTraceCapture will be
// captured and serialized to a DistributedTrace buffer.
uint32_t BeginDistributedTraceCapture(CommandQueueHandle cq);

// Stop capturing the trace and serialize all Fast Dispatch commands
// to a DistributedhBuffer.
void EndDistributedTraceCapture(CommandQueueHandle cq, const uint32_t cq_id);

// Replay the specified trace through the specified CQ.
void EnqueueDistributedTrace(CommandQueueHandle cq, uint32_t trace_id, bool blocking);

// Destroy any metadata/buffers associated with this DistributedTrace
void ReleaseDistributedTrace(std::shared_ptr<MeshDevice> mesh_device, uint32_t trace_id);
```

##

## 4.7 Command Serialization <a id="command-serialization"></a>

This layer captures the serialization framework/protocol used to binarize the commands enqueued into the Virtual Command Queue. The controller interfaces with a virtual mesh and gets lowered into messages that need to get transmitted to each of the RemoteExecutors managing its local mesh of accelerators.

The intention is to design a set of APIs to do the command serialization while being flexible to benchmark a number of library implementations. Initially we elect to go with a schema-based solution that allows for portable, fast implementation and schema evolution of the command set. The plan is to adopt and evaluate flatbuffers which will also allow some interoperability with tt-mlir.

The initial specification for the Command ISA targets the set of host APIs that need to be forwarded from the DistributedDevice to the RemoteExecutor to dispatch onto its local accelerators. The initial command set include:

```cpp
enum class Command {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    ENQUEUE_PROGRAM,
    ENQUEUE_TRACE,
    ENQUEUE_RECORD_EVENT,
    ENQUEUE_WAIT_FOR_EVENT,
    ENQUEUE_SET_RUNTIME_ARGS,
    ENQUEUE_FINISH
};
```

## 4.8 Network Transport <a id="network-transport"></a>

The “Network Transport” layer defines the mechanism for sending serialized commands to remote hosts.

The intention is to design a set of APIs that allow us to swap alternative library implementations for issuing the dispatch based on performance.

We’ve done some evaluation on ZeroMQ, Ray-CPP, nanobind, grpc, as candidates for off-the-shelf library implementations, but this requires further evaluation.

ZeroMQ:

* In an evaluation of ZeroMQ, we’ve been able to reproduce performance benchmarks on a 100Gb Ethernet.

![](images/image041.png)


Ray:

* Limited C++ Language Support: While Ray primarily supports Python, it has experimental support for C++: <https://docs.ray.io/en/latest/ray-overview/installation.html#install-ray-c>
* Provides a lot of useful high-level facilities that are needed for building a distributed platform: handshaking, initialization, console logging.
* Performance benchmarking not done and unclear.

## 4.8 Multithreaded Distributed Runtime Layer

Non-blocking workload and data dispatch across servers requires multi-threading on the Controller Host. In this section, we describe a runtime framework built on top of the resolution, serialization and networking layers; and a multi-threaded approach that enables non-blocking dispatch to Remote Executors interfacing directly with a Physical Mesh of TT-Accelerators.

The diagram below provides a high-level view of the runtime layer, with each pipeline stage running in a different thread. **This is an example configuration and may not be the final solution. The exact assignment of threads will be based on benchmarks.** The goal is to increase pipelining while ensuring that the host is not overloaded.

Components corresponding to the forward path (Controller Main Thread -> Physical Mesh) are displayed in black and the reverse path (Physical Mesh -> Controller Main Thread) is displayed in red.

![](images/image042.png)

### 4.8.1 Message Passing

Message passing between threads on the same host is done through a queue. Host to Host communication is managed by the RPC Sender and Receiver threads interfacing with the networking stack.

In this example, ØMQ Publisher and Subscriber sockets are used, which support broadcasting messages across receivers. For cases where identical data-movement or compute requests need to be sent to multiple Remote Executors, broadcast functionality is useful. Depending on how the Virtual to Physical Resolution layer is implemented (resolution on Controller vs Receiver), messages across hosts may always be broadcasted, or may be a mix of broadcasts and unicasts.

### 4.8.2 Main Thread on the Controller

TT-Distributed users communicate directly with the main thread running on the Controller Host. This is where users submit allocation, compute and data-movement requests. For the prototype, the main thread only populates Host state, i.e. it is responsible for creating MeshWorkload objects and performing memory allocations. Device state is configured on the Event-Loop Thread running on the Remote Executors.

#### 4.8.2.1 Distributed Allocator

The main thread interfaces directly with a Distributed Allocator, managing memory state across physical devices and remote executors in the Virtual Mesh. In this configuration, this allocator is used for all allocations performed in the User-Space: explicit DistributedBuffer or Tensor allocations requested by the user. The Controller is not exposed to memory used for data-paths that perform allocations implicitly (lowering workloads or traces to the physical devices). This memory is managed directly on the Remote Executors. Thus, a user can explicitly configure distributed memory regions to be accessible to the Controller vs the Remote Executors.

It is desirable for the Distributed Allocator, managing User-Space data-structures, to be on the same host as the thread populating distributed workloads. This is because workload creation requires buffer addresses to be passed in as kernel arguments. It is cheaper to query these addresses when they are available locally, rather than querying them through RPC calls, when they are computed on another host. Thus, when workloads are populated on the Controller, the Distributed Allocator will also be on the controller. **This is the case for the POC.**

Alternate configurations allowing all memory regions to be managed on the Controller or the Executors can be explored in different iterations of this framework. Potential solutions consist of the following (depending on the separation of work between the Controller and the Remote Executors):

* **All memory regions managed on the Controller (Executors submit requests):** Remote Executor sends an allocation request to the Controller when performing allocations for Kernel Binaries. Requires a dedicated handler on the Controller to service allocation requests from the Remote Executors.
* **All memory regions managed by the Controller:** Dispatch Command generation + compilation done on the Controller. Thus, the controller manages memory exposed to users and to Runtime data-paths (ex: buffer allocations for Kernel Binaries). The Remote Executors are only responsible for copying pre-assembled Dispatch Commands onto the physical devices they manage.

#### 4.8.2.2 Synchronization and Event Handling on the Controller

Dedicated reader threads are used for event-handling and synchronization on the Controller Hosts. Functionally, these are identical to the *Completion Queue Reader* threads described in the TT-Mesh section.

To issue a blocking operation, the Main Thread on the controller will wait on an event notification from each Remote Executor. This will be done by stalling until the *Completion Queue Reader* updates an event counter for each Executor.

The Executors themselves send notifications through their local *Completion Queue* Readers when event completions are notified by the physical devices. This is done through the RPC Sender Threads on the Executors communicating with the Receiver on the Controller. The Receiver acts as a proxy for the Executors and updates an Event Notification Table/Queue directly on the controller. Once all event entries have been updated, the *Reader* on the Controller unblocks the Main Thread.

### 4.8.3 TT-Mesh Runtime Threads on the Remote-Executors

Remote-Executor hosts accept commands from the Controller and communicate directly with the accelerator-grid. Depending on the distribution of work between the Controller and Executors, it may be necessary to setup a multi-threaded pipeline on the latter. Executors may need to compile Programs, assemble dispatch commands, and perform TMs, all of which are CPU bound operations.

The thread-pool concept introduced in [this](#_3.9.2_Proposed_Design) section will be reused to expand the throughput of TT-Metal native operations on the Executors.

Additionally, an RPC Receiver Thread, performing Network Transfer and Command Deserialization will be added to these hosts. This thread can be considered the main dispatch thread, that passes deserialized commands to an Event Loop Handler Thread (analogous to a worker thread). The Event Loop Handler has access to a thread-pool that can be used to parallelize Host side compute.

Similarly on the reverse path, a reader thread interfaces with an RPC Sender Thread to pass output data and events back to the Controller through the Serialization and Network Transport layers.

**The exact configuration of threads on the Remote Executors will be determined based on micro-benchmarking. The description above assumes that all pipeline stages are in different threads.**

## 4.9 Design Prototype and Scope

1. **Program Compile:**
   1. Offline program compilation feature is pending for Q4’24, so we cannot fully serialize on the controller side. The current scope pushes the work of program/kernel compile to the RemoteExecutor.
   2. In the future we will have support for doing everything on the controller-side.
2. **Host-to-host Data Transfers**:
   1. The initial implementation will involve sending data to RemoteExecutor’s host malloc’d memory.
   2. In the future, we will have an ability to pin host memory and issue the write directly onto device without incurring extra copies. Q1’25 UMD refactor will allow us to pin host-memory.
3. **DistributedAllocator:**
   1. Since program compilation happens in the RemoteExecutor, memory allocations for program binaries are also done on RemoteExecutor. We provision an allocatable region that is managed by the RemoteExecutor.
4. **Llama405b using 2/4 Galaxies Systems:**
   1. With 405B parameters and conservatively 1-byte / parameter (bfp8\_b), and ignoring activation data, we’d need at least two galaxies to get this functional. For performance/accuracy, we’d likely need four galaxy systems.
   2. Initial target will focus on bringing up support for this model end-to-end
   3. We target a RemoteExecutor per remote host that

# 5. TT-NN Integration <a id="tt-nn-integration"></a>

The scope of work involved in TT-NN involves mapping directly to TT-Mesh and TT-Distributed. TT-NN users are mostly transparent to changes underneath the API.

| TT-NN Integration With Multi-Device Native Op Dispatch | | |
| --- | --- | --- |
|  | Single Process | Multi-Process/ Multi-Host |
| Synchronous | Phase I (TT-Mesh) | Phase II (TT-Distributed) |
| Asynchronous | Phase III  (TT-Mesh Performance mode) | Phase IV (TT-Distributed Performance mode) |

Phase I and Phase III will be required to fully deprecate the existing TT-NN Mesh Virtualization. The four diagram illustrates the four stages involved with the integration.

The diagram below shows the different stages of integration:

![](images/image043.png)

While integration work is ongoing, there is also feature-work and bug fixes that require support for tt-train and tt-mlir teams. The category of work involves Python/C++ API parity in TT-NN for multi-device and outstanding feature support.

## 5.1. TT-Mesh/TT-Distributed Interoperability Layer <a id="tt-mesh-tt-distributed-interoperability"></a>

### 5.1.1 What Changes?

1. **Common Interface between Device/MeshDevice**

The existing usage of MeshDevice is already used as like a Device-Like object, but there’s no explicit interface via static or runtime/virtual polymorphism linking these two classes. From a python point of view, duck typing hides this. On the C++ side, it does not.

The proposed changes will leverage the work refining TT-Metalium APIs where opaque DeviceHandle will be exposed instead of concrete Device objects. We introduce explicit TT-Mesh Host APIs for MeshDevice creation, and this will return a handle back to the user. On the implementation side, the MeshDevice will use a version of C++ concepts to impose an interface common to both Device and MeshDevice to give a true Device-Like interoperability.

This will be done with completion of Phase 1.

**2. TTNN Multi-Threaded Runtime:**

For dispatching a SPMD-operation onto N-devices, this currently initiate a program compile on N device worker threads. There is also a program cache per-device. With the proposed changes, there will no longer be a worker thread pinned per device. TT-NN will interface directly with native mesh-level APIs that will orchestrate a single program compile that will be broadcast to devices across the entire mesh.

The plan is to remove *launch_op* entirely from TT-NN. Please see section “2.1 Virtualization Through TTNN” for more context and motivation. We currently implement the virtualization layer in TT-NN, and the asynchronous dispatch of host/device work relies on launch_op to push the work onto executor threads. Instead, pushing work onto a mesh of devices involves enqueueing a MeshWorkload as the unit of compute for the mesh.

This will be done with completion of Phase 1 and 3.

**3. Introduce MeshTensor**

The plan is to remove StorageType variants for MultiDeviceHostStorage and MultiDeviceStorage. Instead, it will interface directly with a MeshBuffer. See Section 3.3 Memory Management: MeshBuffer and MeshAllocator. We’ll explicitly create a construct for a MeshTensor. This will be done with completion of Phase 1 and 3.

**3. Changes from an Op Writer Perspective:**

*“I’m an op-writer – what does this mean for me?”.*

If you are an op-writer that was largely transparent to multi-device, there will be minimal changes besides *launch\_op* going away (if you’re still implementing the op via old-style op-infrastructure).

The new changes will permit an op-writer to program not only to a core mesh, but also to a mesh of devices. Buffer allocation is now guaranteed to occur in lock-step across all devices. See section “3.3: Memory Management” and “3.4 MeshWorkload: Overview, Data-Structures".

An op-writer is not *required* to write an implementation for a MeshWorkload. Many operations in a multi-device context follow an SPMD paradigm (e.g. op-writer focuses on writing a device-local operation, and TT-NN has infrastructure to broadcast this operation across every device in the mesh). For this common use-case, we will supply scaffolding to preserve this behavior.

We will provide guidance on specific changes relevant for an op-writer when the support lands in TT-NN.

**4. Model Writer / TTNN User Perspective:**

Minimal impact. Model writers can largely be transparent to these changes from a TT-NN API usage point of view. Much of the change discussed happens underneath the APIs.

Please leave further questions/comments feedback as review comments here.

# Appendix <a id="appendix"></a>

## Appendix A: Existing Runtime Architecture Overview <a id="appendix-a"></a>

### A.1 Current State of Metal Runtime <a id="appendix-a-current-state-of-runtime"></a>

We present the current components of the Metal Runtime Software stack, and their interactions, in the diagram below. *Note: The arrows in this diagram do not display dataflow, rather, dependencies between components*.

![A diagram of a software](images/image044.png)

Objects/Modules labelled in white exist independently outside the Device class, while objects such as the Firmware, Dispatch and Fabric Management modules are part of the device class.

All Host-Side APIs directly access Command Queue objects (tied to a particular device) to place and/or execute objects such as Buffers, Programs and Traces on TT-Devices using Fast Dispatch.

This diagram does not display Host Runtime infrastructure used by layers of the stack beyond the Host APIs, including Program Cache and Software Command-Queue threads.

**TODO: Describe how device init and management is done today and explain limitations of JIT Build that cause it to be inside a device + why we want to move it outside.**

### A.2 A More Modular Approach to Metal Runtime <a id="appendix-a-more-modular-approach"></a>

In the diagram below, we present a modified architecture for Metal Runtime that places initialization and management modules in a dedicated Control Plane. The JIT Build module, responsible for generating Kernel and Firmware binaries; is similarly a dedicated service that can be used to compile device firmware, Dispatch (and TT-Fabric) kernels and user kernels (embedded in a program). This is similar to the tt\_cluster and UMD layers that get independently initialized and can be used by Device objects to interact with TT-Devices.

![A diagram of a computer](images/image045.png)


This makes the Device object a light-weight data mover (through Command Queues) and memory-manager (through the allocator).

In the existing implementation, all runtime data structures (programs, events, traces and buffers) are constructed based on the unique TT-Device they are meant to reside on. For example, buffers are assigned an address by a device specific allocator. Similarly, physical data or Fast Dispatch commands encoded in programs/traces are derived based on the physical harvesting configuration of each device.

Thus, a Distributed Runtime interfacing with a mesh of devices requires a virtualization layer abstracting away SOC specific details such as harvesting. This will enable users to describe workloads residing on entire meshes, which can be efficiently dispatched using TT-Fabric broadcast operations.

**TODO: Add section on how device class can request control plane to initialize and manage FW based on system level specifications. How this can be extended to setup different FW for the distributed entry point.**

### A.4 Distributed Metal Runtime Using Existing Components and Virtualization <a id="appendix-a-distributed-metal-runtime-using-existing-components-and-virtualization"></a>

Extending the architecture described in section 1.2, the diagram below describes a Distributed Runtime framework with data structures that can span multiple physical TT-Devices.

![A diagram of a software](images/image046.png)

The goal is to abstract all physical device attributes (physical device/core coordinates, dispatch infra and control plane details) through the Virtualization Layer and a dedicated Control Plane initializing and maintaining a Mesh. The MeshDevice class is then an interface to query Mesh Attributes, manage memory on the Mesh (the Lock-Step Allocator manages a distributed memory state described below) and perform Data-Movement through Distributed Fast Dispatch Interfaces.

The sections below describe the architecture of each Distributed/Mesh runtime object, as well as a distributed dispatch layer.

## Appendix B: UMD <a id="appendix-b"></a>

### Appendix B.1: UMD Queries <a id="appendix-b-umd-queries"></a>

```cpp
// ================= These APIs use Predefined Configuration Files =================

// Get the number of DRAM channels on this device. Queries the SOC Descriptor.
int Device::num_dram_channels() const;

// Get L1 size for Tensix worker cores on this device. Queries the SOC Descriptor.
uint32_t Device::l1_size_per_core() const;

// Get the DRAM channel size on this device. Queries the SOC Descriptor.
uint32_t Device::dram_size_per_channel() const;

// Get the DRAM grid size for this device. Queries the SOC Descriptor.
CoreCoord Device::dram_grid_size() const;

// ================ These APIs use Rely on UMD and Physical Devices ================

// Get the SOC Grid Size for this device. Queries SOC Descriptor + harvesting info.
CoreCoord Device::grid_size() const;

// Get the Tensix Grid Size for this device. Queries SOC Descriptor + harvesting info + system type.
CoreCoord Device::logical_grid_size() const;

// Get the worker core grid-size on this device. Queries SOC Descriptor + harvesting info + system type + core descriptor.
CoreCoord Device::compute_with_storage_grid_size() const;

// Get the number of worker cores on this device. Queries SOC Descriptor + harvesting info + system type + core descriptor.
uint32_t Device::num_worker_cores() const;

// Get the active eth cores on this device. Relies on UMD for the cluster descriptor
std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const;

// Check if an eth core is active. Relies on UMD for the cluster descriptor.
bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const;

// Get the inactive eth cores on this device. Relies on UMD for the cluster descriptor
std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const;

// Check if an eth core is inactive. Relies on UMD for the cluster descriptor.
bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const;

// Get the chip and coords of the ethernet core connected to the input.
std::tuple<chip_id_t, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const;

// Get the ethernet sockets on this device that are connected to the input chip_id
std::vector<CoreCoord> Device::get_ethernet_sockets(chip_id_t connected_chip_id) const;

// Check if the device is accessible over MMIO.
bool Device::is_mmio_capable() const;

// ======== These APIs use tt_cluster directly. Not exposed through Device ========

// Get the SOC descriptor for this chip
const metal_SocDescriptor& tt_cluster::get_soc_desc(chip_id_t chip) const;

// Get harvesting information for this chip
uint32_t tt_cluster::get_harvesting_mask(chip_id_t chip) const;

// Get the clock frequency for this chip
int tt_cluster::get_device_aiclk(const chip_id_t& chip_id) const;

// Get the MMIO mapped chip connected to the input device id
chip_id_t tt_cluster::get_associated_mmio_device(chip_id_t device_id) const;

// Get the assigned Host Memory channel for this device
uint16_t tt_cluster::get_assigned_channel_for_device(chip_id_t device_id) const;

// Get the number of host channels for this device
uint32_t tt_cluster::get_num_host_channels(chip_id_t device_id) const;

// Get the host channel size for this device
uint32_t tt_cluster::get_host_channel_size(chip_id_t device_id, uint32_t channel) const;

// Get the architecture for this cluster
ARCH tt_cluster::arch();

// Get the product type for this device
BoardType tt_cluster::get_board_type(chip_id_t chip_id) const;

// Check if the cluster is a galaxy system
bool tt_cluster::is_galaxy_cluster() const;

// Get chip id to ethernet coord map
unordered_map<chip_id_t, eth_coord_t> tt_cluster::get_user_chip_ethernet_coordinates() const;

// Get the number of devices in the Physical Cluster
size_t tt_cluster::number_of_devices();

// Get the number of MMIO mapped/pcie devices in the Physical Cluster
size_t tt_cluster::number_of_pci_devices() const;
```
