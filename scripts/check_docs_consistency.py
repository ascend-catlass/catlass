#!/usr/bin/env python3
# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
"""Forward to the catlass-docs-consistency skill entry point.

The implementation lives with the skill so that it can be installed and reused
outside of this checkout; this file only keeps the repository-relative command
stable for developers.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / ".agents/skills/catlass-docs-consistency/scripts/check_docs.py"

if not TARGET.is_file():
    sys.exit(f"error: documentation check entry point not found at {TARGET}")

os.execv(
    sys.executable,
    [sys.executable, str(TARGET), "--repo-root", str(ROOT), *sys.argv[1:]],
)
