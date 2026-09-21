# AtlasA2 硬件基础信息

## 编程模型

<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas\_ascendc\_10\_00028.html>

## 架构图

<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas\_ascendc\_10\_0011.html>

## 内存单元

| 层级      | 层级细分 | 空间                             | 说明                                                                                                |
| --------- | -------- | -------------------------------- | --------------------------------------------------------------------------------------------------- |
| GM        |          |                                  |                                                                                                     |
| L2 Cache  |          |                                  |                                                                                                     |
| L1 Buffer |          | 512KB                            | 暂存AICore内部反复使用数据，最小访问粒度cacheline-32B，通常切为4个128KB，L1A pingpong，L1B pingpong |
| L0 Buffer | L0A/L0B  | 分别64KB<br />一般各分为两个32KB | Cube的输入<br />通常切分为4个32KB， L0A pingpong，L0B pingpong                                      |
|           | L0C      | 128KB                            | Cube输出，<br />可显式设置doublebuffer，一般通过unit-flag开启边算边搬                               |
|           | BT       | 1KB                              | 存放bias                                                                                            |
|           | FB       | 7KB                              | 存放量化参数，relu参数                                                                              |
| UB        |          | 192KB<br />一般分为两个96KB      | Vector存储                                                                                          |

| 逻辑位置    | 物理位置        | 搬入指令                             | 支持格式                             |
| ----------- | --------------- | ------------------------------------ | ------------------------------------ |
| GM          | GM              | DataCopy                             | RowMajor/ColumnMajor/nZ/zN           |
| A1/B1/C1    | L1              | DataCopy                             | zN(RowMajor)/nZ(ColumnMajor)/ND(m=1) |
| A2          | L0A             | LoadData                             | zZ                                   |
| B2          | L0B             | LoadData                             | nZ                                   |
| CO1         | L0C             | -                                    | zN                                   |
| CO2         | BiasTableBuffer | DataCopy                             | -                                    |
| C2PIPE2GM   | FixpipeBuffer   | DataCopy                             | -                                    |
| tbufVECIN   | UB              | GM->UB DataCopy<br />UB->UB DataCopy | ND/NZ                                |
| tbufVECOUT  | UB              |                                      | ND/NZ                                |
| tbufVECCALC | UB              |                                      | ND/NZ                                |
|             |                 |                                      |                                      |
