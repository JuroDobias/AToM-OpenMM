from __future__ import annotations

from pathlib import Path


class ReceptorNormalizationError(ValueError):
    pass


DEFAULT_RESIDUE_RENAMES = {"CY3": "CYS", "HD1": "HIS", "ZN3": "ZN"}


def normalize_legacy_pdb(source, destination, *, residue_renames=None):
    """Normalize legacy AMBER names while preserving coordinates and TER records."""
    source = Path(source)
    destination = Path(destination)
    renames = dict(DEFAULT_RESIDUE_RENAMES)
    renames.update(residue_renames or {})
    output = []
    removed_extra_particles = 0
    in_chain = False
    for raw in source.read_text().splitlines():
        line = raw
        if line.startswith("TER"):
            if in_chain:
                output.append(line)
                in_chain = False
            continue
        if line.startswith(("ATOM  ", "HETATM")):
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            if atom_name == "EPW" and residue_name in {"WAT", "HOH"}:
                removed_extra_particles += 1
                continue
            replacement = renames.get(residue_name)
            if replacement:
                line = f"{line[:17]}{replacement:>3}{line[20:]}"
            in_chain = True
        output.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "removed_extra_particles": removed_extra_particles,
        "residue_renames": renames,
    }
