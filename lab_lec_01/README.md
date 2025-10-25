# MV-GPU Lab Lecture 01: C++/CUDA Extensions in PyTorch

<!-- An example of writing a C++/CUDA extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial.
This repo demonstrates how to write an example `extension_cpp.ops.mymuladd`
custom op that has both custom CPU and CUDA kernels. -->

This lab guides you through building and using custom C++ and CUDA operators in PyTorch. You will learn how to implement, build, and test a custom op (`extension_cpp.ops.mymuladd`) with HIP support.

For reference, see the official PyTorch tutorial on custom C++ ops.

The examples in this repo work with PyTorch 2.4+.

To build:
```
pip install -r requirement.txt
pip install --no-build-isolation -e .
```

To test:
```
python test/test_custom_op.py
```