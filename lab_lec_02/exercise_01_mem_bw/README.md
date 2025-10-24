# Exercise 01: Measuring Memory Bandwidth

## Setup

```bash
python3 test_pa_v1.py
```

## Lab Guide
Suggested steps:
- Step 0: Try run it first to make sure the program works
- Step 1: Add counters and use profiler to collect memory read and write counters
```bash
rocprofv2 -i counters.txt -d output_profile <your program execution>
```
```
# counters.txt
pmc: FetchSize WriteSize
```
- Step 2: Calculate memory bandwidth based on the counters collected
