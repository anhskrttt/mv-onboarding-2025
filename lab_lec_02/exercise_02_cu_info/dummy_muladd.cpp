#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(expr)                                                                 \
	do                                                                                    \
	{                                                                                     \
		hipError_t _err = (expr);                                                           \
		if (_err != hipSuccess)                                                             \
		{                                                                                   \
			std::cerr << "HIP error: " << hipGetErrorString(_err)                             \
								<< " (" << _err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
			exit(1);                                                                          \
		}                                                                                   \
	} while (0)

__device__ int get_global_cu_id()
{
	unsigned int hw_id = 0;
	asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 0, 32)" : "=s"(hw_id));

	auto wave_id = hw_id & 0xF;
	auto simd_id = (hw_id >> 4) & 0x3;
	auto pipe_id = (hw_id >> 6) & 0x3;
	auto cu_id = (hw_id >> 8) & 0xF;
	auto sh_id = (hw_id >> 12) & 0x1;
	auto se_id = (hw_id >> 13) & 0x7;
	auto tg_id = (hw_id >> 16) & 0xF;
	auto vm_id = (hw_id >> 20) & 0xF;
	auto queue_id = (hw_id >> 24) & 0x7;
	auto state_id = (hw_id >> 27) & 0x7;
	auto me_id = (hw_id >> 30) & 0x3;

	printf("wave_id=%u simd_id=%u pipe_id=%u cu_id=%u sh_id=%u se_id=%u "
				 "tg_id=%u vm_id=%u queue_id=%u state_id=%u me_id=%u\n",
				 wave_id, simd_id, pipe_id, cu_id, sh_id, se_id,
				 tg_id, vm_id, queue_id, state_id, me_id);
	
	// TODO: Complete your formula to compute global CU id here
	// It should be something like this: se_id * <num_CU_per_SE> + cu_id
	return 0;
}

__global__ void dummy_muladd_kernel(
	const float *a, 
	const float *b, 
	const float *c, 
	float *result, 
	int n
	// TODO: Add timestamp array here to save timestamps and load them back to host later to write to file
	// e.g. unsigned long long* timestamps_start, unsigned long long* timestamps_end
)
{
	if (threadIdx.x == 0)
	{
		// TODO: Use get_global_cu_id() to get CU info
		// get_global_cu_id();

		// TODO: Record timestamp at the start of the thread
		// e.g. timestamps_start[blockIdx.x] = wall_clock64();
	}

	// NOTE: Because of 64KB shared mem, each CU can only handle one block at a time
	__shared__ uint8_t shared_buf[65536]; // 64 KB shared memory to make sure one block fits in one CU

	// Use shared memory (dummy way, just to occupy it)
	int shared_idx = (threadIdx.x * 64) % 65536;
	shared_buf[shared_idx] = static_cast<uint8_t>(threadIdx.x);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		// Dummy mod to unbalance workload across blocks
		uint mod = blockIdx.x % 3;
		switch (mod)
		{
		case 0:
			result[idx] = a[idx] * b[idx] + c[idx];

			if (blockIdx.x % 3 == 0)
			{
				// Use a dummy for loop to make this thread busier
				float temp = 0.0f;
				for (int i = 0; i < 99999; i++)
				{
					temp += result[idx] * 0.0001f;
				}

				result[idx] += temp;
			}
			break;
		case 1:
			result[idx] = a[idx] + b[idx] + c[idx];
			break;
		default:
			break;
		}
	}

	// TODO: Record timestamp at the end of the thread
	// e.g. timestamps_end[blockIdx.x] = wall_clock64();
}

int main()
{
	const long long int N = 4096 * 4096 * 16; // Use full 104 active CUs on MI250X
	const size_t size = N * sizeof(float);

	std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 3.0f), h_result(N, 0.0f);

	float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_result = nullptr;

	HIP_CHECK(hipMalloc(&d_a, size));
	HIP_CHECK(hipMalloc(&d_b, size));
	HIP_CHECK(hipMalloc(&d_c, size));
	HIP_CHECK(hipMalloc(&d_result, size));

	// TODO: Allocate timestamp arrays on host/device


	HIP_CHECK(hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_c, h_c.data(), size, hipMemcpyHostToDevice));

	// Launch kernel
	int threads_per_block = 256;
	int num_blocks = (N + threads_per_block - 1) / threads_per_block;

	hipLaunchKernelGGL(dummy_muladd_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
										 d_a, d_b, d_c, d_result, N);

	HIP_CHECK(hipGetLastError());

	HIP_CHECK(hipDeviceSynchronize());

	HIP_CHECK(hipMemcpy(h_result.data(), d_result, size, hipMemcpyDeviceToHost));

	// Print first few results
	for (int i = 0; i < 5; i++)
	{
		std::cout << "result[" << i << "] = " << h_result[i] << std::endl;
	}

	// TODO: Copy back timestamp arrays from device to host, get necessary info and write to file

	HIP_CHECK(hipFree(d_a));
	HIP_CHECK(hipFree(d_b));
	HIP_CHECK(hipFree(d_c));
	HIP_CHECK(hipFree(d_result));

	return 0;
}
