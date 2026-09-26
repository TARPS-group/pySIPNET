# Example data

## `niwot_1999_daily.clim`

Daily meteorological forcing for the Niwot Ridge subalpine forest (US-NR1),
calendar year 1999, used by [`mcmc_calibration.ipynb`](../mcmc_calibration.ipynb).

**Provenance.** Aggregated from the SIPNET project's own example driver at
`sipnet/tests/smoke/niwot/sipnet.clim`, which is byte-identical at the current
pin (`41fa853e`) and at the pin in use when this file was generated, so it does
not need regenerating. The upstream file is sub-daily; the 1999 rows were
aggregated to one record per day:

- `photosynthetically_active_radiation`, `precipitation` — **summed** over each day (totals over the timestep)
- `air_temperature`, `soil_temperature`, `vapor_pressure_deficit`, `soil_vapor_pressure_deficit`, `vapor_pressure`, `wind_speed` — **daily mean**
- `hour_of_day = 0`, `timestep_length = 1.0` (one-day timestep)

The result is a 14-column `.clim` file with 365 daily rows.  (The column count
is the layout discriminator, not a SIPNET version — this SIPNET reads both the
12- and 14-column layouts.)
Pairing this site's meteorology with the nominal Niwot parameters (in the
notebook) gives a physically sensible seasonal NEE cycle — a mid-summer carbon
sink and net release during the colder shoulder seasons (spring/autumn) and
winter.
