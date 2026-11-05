# Epilogue/Tile Class Template Overview

> [Code Location](../../../../../../../include/catlass/epilogue/tile/)

[TOC]

The tile-layer API of the epilogue acts as a template parameter for the epilogue block layer and typically needs to be declared during kernel template assembly. It includes components such as copy, broadcast, element-wise computation (elemwise), type conversion (cast), dequantization (dequant), and swizzle traversal policies.

## API List

### Copy Components

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [copy_gm_to_ub](./copy_gm_to_ub/README.md) | Non-TLA| Atlas A2, Ascend 950 | GM to UB copy (CopyGm2Ub, CopyPerTokenScale2Ub, CopyGm2UbAligned)|
| [copy_gm_to_ub_tla](./copy_gm_to_ub_tla.md) | TLA | Atlas A2, Ascend 950 | GM to UB TLA copy (CopyGm2UbTla)|
| [copy_ub_to_gm](./copy_ub_to_gm/README.md) | Non-TLA| Atlas A2, Ascend 950 | UB to GM copy (CopyUb2Gm, CopyUb2GmAligned)|
| [copy_ub_to_gm_tla](./copy_ub_to_gm_tla.md) | TLA | Atlas A2, Ascend 950 | UB to GM TLA copy (CopyUb2GmTla)|
| [copy_ub_to_l1_tla](./copy_ub_to_l1_tla.md) | TLA | Ascend 950 | UB to L1 copy (zN format)|
| [tile_copy](./tile_copy/README.md) | Composite| Atlas A2, Ascend 950 | Copy composite template (TileCopy, TileCopyBf16, PerTokenDequant, etc.)|

### Broadcast Components

| Component| Style| Description|
| :------ | :------ | :------ |
| [tile_broadcast_add](./tile_broadcast_add.md) | Non-TLA| Row broadcast add (In0 + broadcast(In1))|
| [tile_broadcast_mul](./tile_broadcast_mul/README.md) | Non-TLA + TLA| Broadcast multiply (row broadcast/column broadcast)|
| [tile_broadcast_one_blk](./tile_broadcast_one_blk/README.md) | Non-TLA + TLA| One-block broadcast (scalar → block)|
| [tile_broadcast_inplace_by_column](./tile_broadcast_inplace_by_column.md) | Non-TLA| Column broadcast in-place copy (in-place modification)|
| [tile_broadcast_inplace_by_row](./tile_broadcast_inplace_by_row.md) | Non-TLA| Row broadcast in-place copy (in-place modification)|

### Element-wise Components

| Component| Style| Description|
| :------ | :------ | :------ |
| [tile_elemwise_add](./tile_elemwise_add.md) | Non-TLA| Element-wise add (Add)|
| [tile_elemwise_mul](./tile_elemwise_mul.md) | Non-TLA| Element-wise multiply (Mul)|
| [tile_elemwise_muls](./tile_elemwise_muls.md) | Non-TLA| Element-wise multiply by scalar (Muls)|
| [tile_elemwise_gelu](./tile_elemwise_gelu.md) | Non-TLA| GELU activation function|
| [tile_elemwise_silu](./tile_elemwise_silu.md) | Non-TLA| SiLU/Swish activation function|

### Conversion and Dequantization

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [tile_cast](./tile_cast.md) | Non-TLA| Atlas A2, Ascend 950 | Type conversion (Cast)|
| [tile_pertoken_dequant](./tile_pertoken_dequant.md) | TLA | Ascend 950 | Per-token dequantization (int32 → fp)|

### Swizzle

| Component| Description|
| :------ | :------ |
| [tile_swizzle](./tile_swizzle/README.md) | Tile traversal policy (Identity/Horizontal)|
