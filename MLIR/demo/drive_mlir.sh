#!/bin/bash
LLVMBIN=${HOME}/dev/LLVM/llvm-project/build/ninja_build/bin
LLVMSRC=${HOME}/dev/LLVM/llvm-project/
LLVMLIB=${HOME}/dev/LLVM/llvm-project/build/ninja_build/lib

mlir-opt --convert-std-to-llvm helloworld.mlir  | mlir-cpu-runner --print-module --entry-point-result=i32
#${LLVMBIN}/mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts ${LLVMSRC}/mlir/test/mlir-cpu-runner/sgemm-naive-codegen.mlir | ${LLVMBIN}/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${LLVMLIB}/libmlir_c_runner_utils.dylib
${LLVMBIN}/mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-arith-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts  dgemm.mlir | ${LLVMBIN}/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${LLVMLIB}/libmlir_c_runner_utils.dylib > /dev/null

