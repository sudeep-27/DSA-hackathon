# Budget Traveller with Coupons (Java, single file)

Everything (solver, input handling, CSV output, visualization page) is in `BudgetTraveller.java`. Java 15+.

```
javac BudgetTraveller.java
java BudgetTraveller                                         # menu; after each case: "Add another? (y/n)"
java BudgetTraveller --manual sample_input.txt               # Option 1: one file, many cases
java BudgetTraveller --csv sample_graph.csv 3 2 sample_graph2.csv 1 1   # Option 2: <file> <K> <C> per case
java BudgetTraveller --generate 2 1 --generate 3 2 10 42     # Option 3: <K> <C> [N] [seed] per case
java BudgetTraveller --manual sample_input.txt --csv sample_graph.csv 3 2 --generate 2 1   # mix them
```

Option 1 file format: cases back to back (`N M`, `K C`, then M lines `u v p`). An optional first line with
just `T` gives the number of cases; without it, cases are read until the end of the file.

Outputs: per case, the minimum cost (or -1) on the console and `output_graph_<i>.csv` (path and coupon flights
marked `x`; `output_graph.csv` when there is one case), plus a summary table. `visualization.html` shows every
case with tabs (Play / Next flight / Reset).
