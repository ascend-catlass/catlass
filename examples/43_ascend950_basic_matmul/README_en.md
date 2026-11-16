# 950 Basic Matmul TLA Example Readme

**Note: The community package does not currently support 950 capabilities. Stay tuned for a future supported version.**

## Code Organization

```text
├── 43_ascend950_basic_matmul
│   ├── CMakeLists.txt     # CMake build file
│   ├── README.md
│   └── basic_matmul_tla.cpp # Main file
```

## Usage Example

- After obtaining the code, build the corresponding operator executable. See [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution). This case is a 950 operator, and `-DCATLASS_ARCH=3510` must be added during build.
- Run the operator.

```bash
# Build the specified case
bash scripts/build.sh 43_ascend950_basic_matmul -DCATLASS_ARCH=3510
cd output/bin
# Executable file name | matrix m axis | n axis | k axis | Device ID
# Device ID is optional and defaults to 0
./43_ascend950_basic_matmul 256 512 1024 0
```

The execution result is as follows, indicating that the precision comparison succeeds.

```text
Compare success.
```

## Usage Notes

The DispatchPolicy MmadPingpong used by BasicMatmul by default supports the following template parameters:

| Template Parameter | Default Value | Parameter Description                                                                             |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------- |
| ArchTag            | None          | Specifies the architecture model                                                                  |
| enableUnitFlag     | false         | Specifies whether to enable UnitFlag. It must be set to false when L0C multi-buffering is enabled |
| useHF32            | false         | Specifies whether to enable HF32. Only the float type is supported                                |
| l0CStages          | 1             | Specifies the number of L0C buffers. Set it to 2 to enable L0C double buffering                   |
| enableL1Resident   | false         | Specifies whether to enable L1 residency                                                          |
| l1AStages          | 2             | Number of buffers for loading matrix A on L1                                                      |
| l1BStages          | 2             | Number of buffers for loading matrix B on L1                                                      |
| l0AStages          | 2             | Number of buffers for loading matrix A on L0                                                      |
| l0BStages          | 2             | Number of buffers for loading matrix B on L0                                                      |

Assume the matrix Shape is `M N K`, the tile size on L1 is `m1 n1 k1`, the number of tiles in the M direction is `mTiles = CeilDiv(M, m1)`, the number of tiles in the N direction is `nTiles = CeilDiv(N, n1)`, and the total number of tasks is `taskBlocks = mTiles * nTiles`. enableL1Resident can be enabled in the following two cases:

1. `mTiles = 1`, `nTiles > CoreNum`, and `K < 2 * k1`. In this case, `l0CStages=2` can also be set (enableUnitFlag must be disabled). If there is not enough space and `l0CStages=2` cannot be set, set `n1` to half of the original value.

2. `nTiles = 1`, `mTiles > CoreNum`, and `K < 2 * k1`. In this case, `l0CStages=2` can also be set (enableUnitFlag must be disabled). If there is not enough space and `l0CStages=2` cannot be set, set `m1` to half of the original value.

BasicMatmul also supports DispatchPolicy MmadPreloadAsyncWithCallback, which supports the following template parameters:

| Template Parameter | Default Value | Parameter Description                                                                             |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------- |
| ArchTag            | None          | Specifies the architecture model                                                                  |
| preloadStages      | None          | Specifies the number of preloads                                                                  |
| l1AStages          | 2             | Number of buffers for loading matrix A on L1                                                      |
| l1BStages          | 2             | Number of buffers for loading matrix B on L1                                                      |
| l0AStages          | 2             | Number of buffers for loading matrix A on L0                                                      |
| l0BStages          | 2             | Number of buffers for loading matrix B on L0                                                      |
| l0CStages          | 1             | Specifies the number of L0C buffers. Set it to 2 to enable L0C double buffering                   |
| enableUnitFlag     | false         | Specifies whether to enable UnitFlag. It must be set to false when L0C multi-buffering is enabled |
| enableShuffleK     | false         | Specifies whether to enable K-direction staggered reading                                         |
| useHF32            | false         | Specifies whether to enable HF32. Only the float type is supported                                |
| enableL1Resident   | false         | Specifies whether to enable L1 residency                                                          |

Compared with `MmadPingpong`, `MmadPreloadAsyncWithCallback` has two more template parameters. One is `preloadStages`. This parameter is usually set to 1 and specifies the number of preloads. When this parameter is set to 1, the first loop only loads data and does not perform matmul computation. The second loop first loads the data for the second loop, and then completes the Matmul computation of the previous loop, and so on. After the final loop ends, one additional Matmul computation is performed. The benefit is that the data required for the current Matmul computation has already been moved in the previous loop. Therefore, instruction issue is advanced, which reduces the performance loss caused by instruction issue latency.

The second parameter is `enableShuffleK`. This parameter is mainly used to avoid bandwidth loss caused by same-address access conflicts. The main principle is to stagger the data read addresses of each core. This parameter does not need to be enabled on 950.

Compared with `MmadPingpong`, `MmadPreloadAsyncWithCallback` has more optimization points, but its logic is also more complex and has higher Scalar overhead. Use it based on the scenario, especially for small Shape scenarios.
