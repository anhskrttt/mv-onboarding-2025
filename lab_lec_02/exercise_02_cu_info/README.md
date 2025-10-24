# Exercise 02: Collecting CU Information

## Setup

```bash
hipcc dummy_muladd.cpp -o dummy_muladd
./dummy_muladd
```

## Lab Guide

Suggested steps:

- Step 0: Try run it first to make sure the program work
- Step 1: Find the formula to calculate global cu_id and reverify if it is correct
- Step 2: Modify the code to get
	- global_cu_id
	- each thread's start timestamp and end timestamp
	- calculate each cu's execution time
- Step 3: Write the collected information to a file in the host side and parse it to be able to draw graphs and analyze