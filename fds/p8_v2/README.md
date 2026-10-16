# P8 v2 offline FDS inputs

These inputs define configurable **synthetic references**, not calibrated W1
fires.  The 40.0 m x 12.5 m x 3.0 m domain and all source properties are
simulation assumptions.  Run them offline with the pinned FDS release recorded
by `scripts/run_p8_v2_fds.sh`; raw FDS output is intentionally not committed.

The current Python test and demo pipeline uses `reference_provider =
reduced_order_fixture` until converted FDS snapshots are supplied.  It must not
label those fixtures as FDS data.
