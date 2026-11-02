---
nav_order: 10
---

# Environment Preparation and Build

This directory explains how to prepare the development environment, build CATLASS Python, and run tests.

## Environment Setup

| Document | Scope |
|------|------|
| [Environment Preparation](environment.md) | Overall environment requirements, dependency software, and environment checks. |
| [Conda Installation](conda.md) | Create a development environment with the repository `environment.yml` (does not include CANN or AscendNPU-IR). |
| [Docker Installation](docker.md) | Build a development image with the repository `Dockerfile`, including all development dependencies. |
| [Building AscendNPU-IR](ascend_npu_ir.md) | Manually build the AscendNPU-IR that CATLASS Python depends on. |

## Build and Test

| Document | Scope |
|------|------|
| [Building CATLASS Python](build.md) | `./build.sh` build (Development / Release mode) and artifact checks. |
| [Running Test Cases](testing.md) | pytest, lit test, and end-to-end verification. |
