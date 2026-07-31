# `slipgen.in` File Format

This file is the input to `src/slipgen_von_karman.f90` and is
generated automatically by `write_fault_in()` in `src/run_pipeline.py`.
It is a plain-text, tab-separated file with an inline `!` comment on
each header line (Fortran comment syntax).

```
L      W                 ! Line 1: fault length, width (km)
M      N                 ! Line 2: FFT grid points (strike, dip)
ax     az                ! Line 3: correlation lengths (km)
itype  H                 ! Line 4: itype (1=Von Karman, 2=Exponential, 3=Gaussian), Hurst exponent
NX     NY                ! Line 5: coarse subfault grid dimensions (strike blocks, dip blocks)
<NY lines of NX slip values, tab-separated, in meters>
```

## Field reference

| Field | Meaning | Source |
|---|---|---|
| `L`, `W` | Fault length and width (km) | GPR prediction (`Eff. Length`, `Eff. Width`) |
| `M`, `N` | Fine FFT grid resolution | `grid_N` parameter, passed straight through |
| `ax`, `az` | Fractal correlation lengths along strike/dip (km) | GPR prediction for the chosen spectral model (`ax_von`/`az_von`, `ax_exp`/`az_exp`, or `ax_gauss`/`az_gauss`) |
| `itype` | Spectral model: `1`=Von Kármán, `2`=Exponential, `3`=Gaussian | Set by `model_choice` in `run_pipeline()` |
| `H` | Hurst exponent controlling fractal roughness | `H` parameter |
| `NX`, `NY` | Coarse block-grid dimensions used for asperity placement | Derived from `n_blocks` and the fault aspect ratio (both capped at 5) |
| Slip grid | `NY` rows × `NX` tab-separated slip values (m) | Computed by `write_fault_in()`: background slip plus two nested asperity regions (ALA, AVLA) |

## Asperity regions

- **ALA** (Area of Large Asperity): forced to be a square block region,
  sized so its area is at least the GPR-predicted `Area of large
  asperity`.
- **AVLA** (Area of Very Large Asperity): a smaller rectangular region
  nested *inside* the ALA square, sized to match the GPR-predicted
  `Area of very large asperity` as closely as possible in whole
  blocks.
- **Background**: all remaining blocks, assigned whatever slip value
  keeps the area-weighted average slip across the whole grid equal to
  the GPR-predicted mean slip.

Slip values: `slip_AVLA = mean + 1.0σ`, `slip_ALA = mean + 0.5σ`,
background solved algebraically. If this would force a negative
background slip, `std_slip` is automatically capped and the three
values recomputed (a notice is printed when this happens).

## Fortran program outputs

Running the compiled `slipgen_von_karman.f90` program reads
`slipgen.in` (via Fortran unit 130) and writes:

| File | Unit | Contents |
|---|---|---|
| `slipgen.txt` | 101 | Main output slip field |
| `specx.txt` | 105 | Along-strike spectral diagnostics |
| `specy.txt` | 106 | Along-dip spectral diagnostics |
| `slipx.txt` | 120 | Along-strike slip cross-section |
| `slipy.txt` | 121 | Along-dip slip cross-section |

These are all excluded from version control via `.gitignore` since
they're regenerated outputs, not source.
