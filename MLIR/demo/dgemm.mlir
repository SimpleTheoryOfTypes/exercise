// C += A * B.
func @matmul(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  %t_start = call @rtclock() : () -> f64
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf64>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf64>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf64>
        %p = arith.mulf %a, %b : f64
        %co = arith.addf %ci, %p : f64
        affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf64>
      }
    }
  }
  %t_end = call @rtclock() : () -> f64
  %t = arith.subf %t_end, %t_start : f64

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
 
  %M = memref.dim %C, %c0 : memref<2048x2048xf64>
  %N = memref.dim %C, %c1 : memref<2048x2048xf64>
  %K = memref.dim %A, %c1 : memref<2048x2048xf64>

  %f1 = arith.muli %M, %N : index
  %f2 = arith.muli %f1, %K : index
  
  // 2*M*N*K.
  %f3 = arith.muli %c2, %f2 : index
  %reps = arith.constant 1 : index
  %num_flops = arith.muli %reps, %f3 : index
  %num_flops_i = arith.index_cast %num_flops : index to i16
  %num_flops_f = arith.sitofp %num_flops_i : i16 to f64
  %flops = arith.divf %num_flops_f, %t : f64
  call @print_flops(%num_flops_f) : (f64) -> ()
  call @print_flops(%flops) : (f64) -> ()
 
  call @printF64(%t) : (f64) -> ()
  return
}

func @main() {
  %A = memref.alloc() : memref<2048x2048xf64>
  %B = memref.alloc() : memref<2048x2048xf64>
  %C = memref.alloc() : memref<2048x2048xf64>

  %cf1 = arith.constant 1.0 : f64


  linalg.fill(%cf1, %A) : f64, memref<2048x2048xf64>
  linalg.fill(%cf1, %B) : f64, memref<2048x2048xf64>
  linalg.fill(%cf1, %C) : f64, memref<2048x2048xf64>

  call @matmul(%A, %B, %C) : (memref<2048x2048xf64>, memref<2048x2048xf64>, memref<2048x2048xf64>) -> ()
  return
}

func private @print_flops(f64)
func private @rtclock() -> f64
func private @printF64(f64)
