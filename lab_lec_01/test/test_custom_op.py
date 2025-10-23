import torch
import time

# TODO(Step 5): import your custom C++ extension module here
# import <package_name>

def reference_muladd(a, b, c):
    return a * b + c

def benchmark(fn, *args, n_warmup=10, n_iter=100):
    # Warmup
    for _ in range(n_warmup):
        _ = fn(*args)
    torch.cuda.synchronize() if args[0].is_cuda else None

    # Benchmark
    start = time.time()
    for _ in range(n_iter):
        _ = fn(*args)
    torch.cuda.synchronize() if args[0].is_cuda else None
    end = time.time()

    avg_ms = (end - start) * 1000 / n_iter
    return avg_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Test shapes
    test_shapes = [(1000,), (1024, 1024), (4096, 4096)]

    for shape in test_shapes:
        print(f"\n=== Shape: {shape} ===")
        a = torch.randn(shape, device=device, dtype=torch.float32)
        b = torch.randn(shape, device=device, dtype=torch.float32)
        c = 1.23  # scalar

        # TODO(Step 5.1): Change this to call your custom C++ op hip_mymuladd instead of muladd CPU implementation
        result = reference_muladd(a, b, c)
        
        # Reference result (CPU implementation)
        ref = reference_muladd(a, b, c)

        # Check correctness
        try:
            torch.testing.assert_close(result, ref, atol=1e-5, rtol=1e-5)
            print("Accuracy check: PASSED")
        except AssertionError as e:
            print("Accuracy check: FAILED")
            print(e)

        # Benchmark performance
        cpu_a, cpu_b = a.cpu(), b.cpu()

        cpu_latency = benchmark(reference_muladd, cpu_a, cpu_b, c)
        
        # TODO(Step 5.2): Change this to your custom C++ op for benchmarking
        # Also change to use device tensors instead of CPU tensors
        custom_latency = benchmark(reference_muladd, cpu_a, cpu_b, c)

        print(f"CPU latency:    {cpu_latency:.3f} ms")
        print(f"Custom op latency ({device}): {custom_latency:.3f} ms")
        print(f"Speedup: {cpu_latency / custom_latency:.2f}x")


if __name__ == "__main__":
    main()
