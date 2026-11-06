"""Measure Ascend950 SYRK kernels with NPU events on a dedicated stream."""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401
import torch_catlass


def _dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _measure(run, warmup, iterations):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(warmup):
            run()
    stream.synchronize()

    samples = []
    with torch.npu.stream(stream):
        for _ in range(iterations):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record(stream)
            run()
            end.record(stream)
            samples.append((start, end))
    stream.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in samples)


def _run_case(name, implementation, x, residual, alpha, beta, diagonal, tile, splitk, warmup, iterations):
    if implementation == "torch":
        if name == "basic":
            run = lambda: torch.matmul(x, x.transpose(-2, -1))
        else:
            def run():
                output = torch.matmul(x, x.transpose(-2, -1))
                output.mul_(alpha)
                output.add_(residual, alpha=beta)
                output.diagonal(dim1=-2, dim2=-1).add_(diagonal)
                return output
    elif name == "basic":
        run = lambda: torch_catlass.ascend950_basic_syrk(x, outDType=x.dtype, tile=tile, splitk=splitk)
    else:
        run = lambda: torch_catlass.ascend950_syrk_polynomial(
            x, x, residual, alpha, beta, diagonal, outDType=x.dtype, tile=tile, splitk=splitk
        )
    return _measure(run, warmup, iterations)


def _verify(x, residual, alpha, beta, diagonal, tile, splitk, names):
    product = torch.matmul(x.float(), x.float().transpose(-2, -1))
    if "basic" in names:
        basic = torch_catlass.ascend950_basic_syrk(x, outDType=x.dtype, tile=tile, splitk=splitk)
        torch.testing.assert_close(basic.float(), product, rtol=2e-2, atol=2e-2)

    if "polynomial" in names:
        polynomial = torch_catlass.ascend950_syrk_polynomial(
            x, x, residual, alpha, beta, diagonal, outDType=x.dtype, tile=tile, splitk=splitk
        )
        expected = alpha * product + beta * residual.float()
        expected.diagonal(dim1=-2, dim2=-1).add_(diagonal)
        torch.testing.assert_close(polynomial.float(), expected, rtol=2e-2, atol=2e-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("basic", "polynomial", "all"), default="all")
    parser.add_argument("--implementation", choices=("catlass", "torch"), default="catlass")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--splitk", type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--alpha", type=float, default=2.0315)
    parser.add_argument("--beta", type=float, default=-4.7750)
    parser.add_argument("--diagonal", type=float, default=3.4445)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.m % 32 != 0:
        parser.error("--m must be divisible by 32 for polynomial and split-K cases")
    if args.batch < 1 or args.k < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch, k, and iterations must be positive; warmup must be non-negative")

    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    dtype = _dtype(args.dtype)
    shape = (args.batch, args.m, args.k) if args.batch > 1 else (args.m, args.k)
    x = torch.randn(shape, dtype=dtype, device="npu") * (args.k**-0.5)
    residual = torch.matmul(x, x.transpose(-2, -1))

    names = ("basic", "polynomial") if args.op == "all" else (args.op,)
    splitks = (1,) if args.splitk == 1 else (1, args.splitk)
    if args.verify:
        for splitk in splitks:
            _verify(x, residual, args.alpha, args.beta, args.diagonal, args.tile, splitk, names)

    results = []
    for name in names:
        for splitk in splitks:
            milliseconds = _run_case(
                name,
                args.implementation,
                x,
                residual,
                args.alpha,
                args.beta,
                args.diagonal,
                args.tile,
                splitk,
                args.warmup,
                args.iterations,
            )
            results.append({
                "op": name,
                "implementation": args.implementation,
                "batch": args.batch,
                "m": args.m,
                "k": args.k,
                "dtype": args.dtype,
                "tile": args.tile,
                "splitk": splitk,
                "alpha": args.alpha,
                "beta": args.beta,
                "diagonal": args.diagonal,
                "milliseconds": milliseconds,
            })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
