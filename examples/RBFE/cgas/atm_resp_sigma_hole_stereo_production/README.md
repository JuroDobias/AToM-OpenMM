# cGAS ATM production run

Production NEQTI calculation for the `ms_491 -> ms_492` stereochemical edge.
It uses cached GAFF2/multi-conformer RESP parameters with chlorine sigma holes,
native A/B endpoint REST2, excluded ATM schedule optimization, 100 ps half-path
switches, and convergence-based stopping at up to 100 samples per direction.

The 12-hour Slurm script resubmits itself after a wall-time signal and resumes
from atomic NEQTI and REST2 checkpoints.
