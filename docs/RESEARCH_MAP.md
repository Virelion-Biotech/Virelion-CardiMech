# Research and upstream map

This document records architecture and implementation ideas intentionally mined from the open cardiac mechanics ecosystem. CardiMech does not claim ownership of upstream algorithms and does not silently vendor third-party code.

| Project | Useful pattern for CardiMech | Integration stance |
|---|---|---|
| fenicsx-pulse | separation of material models, boundary/loading description, mechanics problem and solver | preferred optional FEniCSx adapter target |
| pulse | mature predecessor patterns for mechanics BCs and nonlinear solve workflows | architecture/reference; LGPL-3.0 upstream, not vendored |
| simcardems2 | explicit staggered electrophysiology-mechanics coupling | adapter/reference |
| Ambit | multiphysics separation and 0D/solid/FSI coupling | plugin/external backend candidate |
| CardioMechanics | passive/active tissue, reference recovery, parameter optimization and lumped circulation | external backend/reference; license reviewed separately |
| Chaste | strong benchmark/tutorial/test culture for electromechanics | validation/reference target |
| cardiac_benchmark / 2025 elastodynamics benchmark | reproducible passive/active/pericardial mechanics benchmark cases | verification target for future spatial adapters |
| ModularCirc | composable chambers/valves/vessels for 0D circulation | architecture reference for future circulation expansion |
| cardiac-geometriesx | clean solver-ready geometry representation | anatomy/backend interoperability reference |
| fenicsx-ldrb | rule-based myocardial microstructure | consumed upstream through CardiAnatomy rather than duplicated |

## What was deliberately not copied

The built-in NumPy reference mechanics/circulation implementation was written specifically for CardiMech. It is not a transcription of any upstream FEM, circulation, or electromechanics source. The external ecosystem table is for compatibility planning and provenance of architectural ideas.
