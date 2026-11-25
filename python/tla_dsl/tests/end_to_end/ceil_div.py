"""Verify Int32 ceil_div on Ascend 950."""

from __future__ import annotations

import argparse

import catlass.tla as tla


@tla.kernel
def ceil_div_kernel(out: tla.Tensor, a: tla.Int32, b: tla.Int32) -> None:
    out[0] = tla.ceil_div(a, b)
    out[1] = tla.ceil_div(a, 64)
    out[2] = tla.ceil_div(129, b)
    out[3] = tla.ceil_div(a // 2 + 1, b)
    out[4] = tla.ceil_div(2147483647, 2)
    for i in tla.range(5, 7):
        out[i] = tla.ceil_div(i, b)
    chosen = a
    if a < b:
        chosen = b
    out[7] = tla.ceil_div(chosen, b)
    if b != 0:
        out[8] = ((a % b) != 0).to(tla.Int32)
    else:
        out[8] = 0


def main() -> None:
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    out = torch.empty(9, dtype=torch.int32, device=f"npu:{args.device}")
    tensor = tla.from_dlpack(out, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(ceil_div_kernel, tensor, tla.Int32(0), tla.Int32(0))

    def reference(a: int, b: int) -> int:
        return (a + b - 1) // b if b else 0

    cases = (
        (128, 64), (129, 64), (1, 64), (100, 1), (100, 0), (0, 64), (0, 0),
        (2147483647, 1), (2147483647, 2147483647), (2147483647, 2),
    )
    for a, b in cases:
        out.fill_(-1)
        artifact(tensor, tla.Int32(a), tla.Int32(b), block_num=1)
        torch.npu.synchronize()
        expected = [
            reference(a, b),
            reference(a, 64),
            reference(129, b),
            reference(a // 2 + 1, b),
            1073741824,
            reference(5, b),
            reference(6, b),
            reference(max(a, b), b),
            int(a % b != 0) if b else 0,
        ]
        actual = out.tolist()
        if actual != expected:
            raise AssertionError(f"ceil_div inputs ({a}, {b}): expected {expected}, got {actual}")
    print(f"ceil_div verification passed for {len(cases)} cases")


if __name__ == "__main__":
    main()
