#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <hip/hip_runtime.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

namespace extension_cpp
{

  // ============ TODO(Step 1): Add HIP implememtations for `mymuladd_cuda` ============
  // at::Tensor mymuladd_cuda(...) { ... }
   
  // ============================== End of TODO(Step 1) ================================

  __global__ void mul_kernel(int numel, const float *a, const float *b, float *result)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
      result[idx] = a[idx] * b[idx];
  }

  at::Tensor mymul_cuda(const at::Tensor &a, const at::Tensor &b)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    int numel = a_contig.numel();
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(a));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    mul_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
    return result;
  }

  __global__ void add_kernel(int numel, const float *a, const float *b, float *result)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
      result[idx] = a[idx] + b[idx];
  }

  void myadd_out_cuda(const at::Tensor &a, const at::Tensor &b, at::Tensor &out)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    int numel = a_contig.numel();
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(a));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    add_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
  }

  // Defines the operators
  TORCH_LIBRARY(extension_cpp, m)
  {
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
    // TODO(Step 3): Add operator definitions
    
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m)
  {
    m.impl("mymul", &mymul_cuda);
    m.impl("myadd_out", &myadd_out_cuda);
    // TODO(Step 4): Register HIP implementations for mymuladd

  }

}

// TODO: Register the extension (But I already did this part for you)
extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  // NOTE: It's PyInit_<module_name>
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}