# Example data

## `niwot_1999_daily.clim`

Daily meteorological forcing for the Niwot Ridge subalpine forest (US-NR1),
calendar year 1999, used by [`mcmc_calibration.ipynb`](../mcmc_calibration.ipynb).

**Provenance.** Aggregated from the SIPNET project's own example driver at
`sipnet/tests/smoke/niwot/sipnet.clim` (pinned submodule commit
`e4abf14f2445133c785b756025a2e39e60c7760f`). The upstream file is sub-daily; the
1999 rows were aggregated to one record per day:

- `par`, `precip` — **summed** over each day (totals over the timestep)
- `tair`, `tsoil`, `vpd`, `vpd_soil`, `vpress`, `wspd` — **daily mean**
- `time = 0`, `length = 1.0` (one-day timestep)

The result is a standard 14-column SIPNET v1 `.clim` file with 365 daily rows.
Pairing this site's meteorology with the nominal Niwot parameters (in the
notebook) gives a physically sensible seasonal NEE cycle — a mid-summer carbon
sink and net release during the colder shoulder seasons (spring/autumn) and
winter.
