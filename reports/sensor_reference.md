# CMAPSS Sensor & Operational Setting Reference

Source: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, 2008.
(Publicly documented alongside the NASA CMAPSS dataset.)

## Operational Settings (3)

| Column | Description |
|---|---|
| op_setting_1 | Altitude |
| op_setting_2 | Mach number |
| op_setting_3 | Throttle Resolver Angle (TRA) |

In FD001 and FD003 these are nearly constant (single operating condition).
In FD002 and FD004 they vary across 6 discrete operating regimes.

## Sensor Measurements (21)

| Column | Symbol | Description | Units |
|---|---|---|---|
| sensor_1  | T2    | Total temperature at fan inlet | °R |
| sensor_2  | T24   | Total temperature at LPC outlet | °R |
| sensor_3  | T30   | Total temperature at HPC outlet | °R |
| sensor_4  | T50   | Total temperature at LPT outlet | °R |
| sensor_5  | P2    | Pressure at fan inlet | psia |
| sensor_6  | P15   | Total pressure in bypass-duct | psia |
| sensor_7  | P30   | Total pressure at HPC outlet | psia |
| sensor_8  | Nf    | Physical fan speed | rpm |
| sensor_9  | Nc    | Physical core speed | rpm |
| sensor_10 | epr   | Engine pressure ratio (P50/P2) | -- |
| sensor_11 | Ps30  | Static pressure at HPC outlet | psia |
| sensor_12 | phi   | Ratio of fuel flow to Ps30 | pps/psi |
| sensor_13 | NRf   | Corrected fan speed | rpm |
| sensor_14 | NRc   | Corrected core speed | rpm |
| sensor_15 | BPR   | Bypass ratio | -- |
| sensor_16 | farB  | Burner fuel-air ratio | -- |
| sensor_17 | htBleed | Bleed enthalpy | -- |
| sensor_18 | Nf_dmd | Demanded fan speed | rpm |
| sensor_19 | PCNfR_dmd | Demanded corrected fan speed | rpm |
| sensor_20 | W31   | HPT coolant bleed | lbm/s |
| sensor_21 | W32   | LPT coolant bleed | lbm/s |

## Notes for EDA (Phase 2)

- HPC = High Pressure Compressor, LPC = Low Pressure Compressor,
  HPT = High Pressure Turbine, LPT = Low Pressure Turbine.
- In FD001 (HPC degradation only), expect the clearest degradation trends
  in sensors tied to HPC: **sensor_3 (T30), sensor_7 (P30), sensor_11 (Ps30)**.
- Several sensors are known to be **near-constant / non-informative** in
  FD001 (e.g. sensor_1, sensor_5, sensor_6, sensor_10, sensor_16,
  sensor_18, sensor_19) — this is something EDA should confirm empirically
  rather than assume, and is a common candidate talking point: "I verified
  which sensors carry signal rather than trusting a blog post."