# Exam Timetable Scheduling

Pure Python 3, no packages to install.

```
python timetable.py --input sample_input.txt              # Option 1: file (N M R + M lines "u v")
python timetable.py --random -n 20 -r 5 --density 0.3     # Option 2: random graph
python timetable.py                                       # interactive menu
```
Open `report.html` in any browser. It contains all three outputs, the coloured conflict graph,
the timetable grid, the assignment animation (Play/Step) and the comparison charts as N grows.
The CSV (Output 3) downloads with the button in the report; add `--csv` to also write CSV files.

Other flags: `--out report.html`, `--time-limit 10`, `--no-benchmark`, `--selftest`.
