| Case | Cycles | Latency (ns) | Off-chip bytes | PIM-local bytes | Peak SRAM | DRAM allocation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| conventional_python_unoptimized | 1169 | 1169 | 19984 | 0 | 4608 | 9536 |
| conventional_python_optimized | 945 | 945 | 13840 | 0 | 4096 | 8768 |
| conventional_mlir_unoptimized | 1169 | 1169 | 19984 | 0 | 4608 | 9536 |
| conventional_mlir_optimized | 945 | 945 | 13840 | 0 | 4096 | 8768 |
| pim_python_unoptimized | 1022 | 1022 | 15760 | 4224 | 4608 | 9536 |
| pim_python_optimized | 798 | 798 | 9616 | 4224 | 3072 | 8768 |
| pim_mlir_unoptimized | 1022 | 1022 | 15760 | 4224 | 4608 | 9536 |
| pim_mlir_optimized | 798 | 798 | 9616 | 4224 | 3072 | 8768 |
