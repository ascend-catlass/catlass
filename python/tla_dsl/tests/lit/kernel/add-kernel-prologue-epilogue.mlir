// RUN: if %tla_compile --print-pipeline=mlir 2>&1 | grep -q add-kernel-prologue-epilogue; then %tla_compile %s -o - | FileCheck %s --check-prefix=HIVM; else echo "add-kernel-prologue-epilogue unavailable" | FileCheck %s --check-prefix=NOHIVM; fi

module {
  func.func @kernel() {
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// HIVM-LABEL: func.func @kernel
// Plain HIVM funcs without HACC entry attrs are skipped by the prologue pass.
// HIVM-COUNT-1: hivm.hir.pipe_barrier[<PIPE_ALL>]
// HIVM: return

// NOHIVM: add-kernel-prologue-epilogue unavailable
