# data/

Place your input files here before running the pipeline.

## Required

| File | Description |
|------|-------------|
| `meta_info.xlsx` | Excel workbook with one sheet per species/cell-type group. Each sheet lists NWB filenames and AD channels (e.g. `AD0`, `AD2`). Sheet names are used to infer species (`mouse`, `NHP`, `human`) and cell type (E / I). |
| `*.nwb` (or subdirectories) | MIES multipatch NWB files (v1 or v2). Subdirectory nesting is fine — the script recurses. |

## Notes

- NWB files are not included in this repository because they are large binary files.
- The `meta_info.xlsx` filename is the default; edit `EXCEL_PATH` at the top of `extract_intrinsic.py` if yours differs.
- The pipeline detects NWB v1 and v2 automatically and monkey-patches neuroanalysis for oodDAQ compatibility.
