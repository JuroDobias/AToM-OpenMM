from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm as mm
from openff.toolkit import ForceField, Molecule
from openff.units import unit as offunit


class CovalentParameterError(RuntimeError):
    pass


@dataclass(frozen=True)
class CovalentParameterBundle:
    molecule: Molecule
    system: mm.System
    charges_e: np.ndarray
    cache_key: str
    provenance: dict[str, object]


def _canonical_cache_key(molecule: Molecule, model: str, forcefield: str) -> str:
    payload = {
        "mapped_isomeric_smiles": molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        ),
        "formal_charge": int(round(molecule.total_charge.m_as(offunit.elementary_charge))),
        "charge_model": model,
        "forcefield": forcefield,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _predict_espaloma_charges(molecule: Molecule, model: str) -> np.ndarray:
    try:
        import espaloma as esp
        from openmmforcefields.generators import EspalomaTemplateGenerator
    except ImportError as exc:
        raise CovalentParameterError(
            "Espaloma NN charges were requested, but espaloma and openmmforcefields are not installed"
        ) from exc

    generator = EspalomaTemplateGenerator(
        forcefield=model,
        template_generator_kwargs={
            "reference_forcefield": "openff_unconstrained-2.2.1",
            "charge_method": "nn",
        },
    )
    graph = esp.Graph(molecule)
    generator.espaloma_model(graph.heterograph)
    try:
        charges = graph.nodes["n1"].data["q"].flatten().detach().cpu().numpy()
    except Exception as exc:
        raise CovalentParameterError("Espaloma did not produce atomic NN charges") from exc
    return np.asarray(charges, dtype=np.float64)


def constrain_charge_sum(
    charges_e: np.ndarray,
    adjustable_indices: list[int] | np.ndarray,
    target_total_e: float,
) -> tuple[np.ndarray, float]:
    charges = np.asarray(charges_e, dtype=np.float64).copy()
    adjustable = np.asarray(adjustable_indices, dtype=int)
    if adjustable.size == 0:
        raise CovalentParameterError("charge correction requires at least one adjustable atom")
    correction = (float(target_total_e) - float(charges.sum())) / float(adjustable.size)
    charges[adjustable] += correction
    if not np.isclose(charges.sum(), target_total_e, atol=1.0e-10):
        raise CovalentParameterError("charge correction did not reach the requested total")
    return charges, correction


def assign_espaloma_nn_charges(
    molecule: Molecule,
    *,
    cache_dir: Path | None = None,
    model: str = "espaloma-0.3.2",
    forcefield: str = "openff-2.2.1.offxml",
) -> tuple[np.ndarray, str, bool]:
    cache_key = _canonical_cache_key(molecule, model, forcefield)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{cache_key}.charges.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            charges = np.asarray(data["charges_e"], dtype=np.float64)
            if charges.shape != (molecule.n_atoms,):
                raise CovalentParameterError(f"cached charges in {cache_file} have the wrong size")
            return charges, cache_key, True

    charges = _predict_espaloma_charges(molecule, model)
    if charges.shape != (molecule.n_atoms,):
        raise CovalentParameterError(
            f"Espaloma returned {charges.size} charges for {molecule.n_atoms} atoms"
        )
    formal_charge = float(molecule.total_charge.m_as(offunit.elementary_charge))
    charges, _ = constrain_charge_sum(charges, np.arange(molecule.n_atoms), formal_charge)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cache_key": cache_key,
                    "charge_model": "espaloma_nn",
                    "espaloma_model": model,
                    "forcefield": forcefield,
                    "charges_e": charges.tolist(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return charges, cache_key, False


def _system_charges(system: mm.System) -> np.ndarray:
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        raise CovalentParameterError(
            f"expected one OpenFF NonbondedForce, found {len(forces)}"
        )
    force = forces[0]
    return np.asarray(
        [
            force.getParticleParameters(index)[0].value_in_unit(mm.unit.elementary_charge)
            for index in range(force.getNumParticles())
        ],
        dtype=np.float64,
    )


def parameterize_capped_product(
    sdf: Path,
    *,
    cache_dir: Path | None = None,
    forcefield: str = "openff-2.2.1.offxml",
    espaloma_model: str = "espaloma-0.3.2",
) -> CovalentParameterBundle:
    molecule = Molecule.from_file(str(sdf), allow_undefined_stereo=False)
    charges, cache_key, cache_hit = assign_espaloma_nn_charges(
        molecule,
        cache_dir=cache_dir,
        model=espaloma_model,
        forcefield=forcefield,
    )
    molecule.partial_charges = charges * offunit.elementary_charge
    openff = ForceField(forcefield)
    system = openff.create_openmm_system(
        molecule.to_topology(),
        charge_from_molecules=[molecule],
        allow_nonintegral_charges=False,
    )
    observed = _system_charges(system)
    if not np.allclose(observed, charges, atol=1.0e-7):
        raise CovalentParameterError(
            "OpenFF system charges differ from the supplied Espaloma NN charges"
        )
    provenance = {
        "charge_model": "espaloma_nn",
        "espaloma_model": espaloma_model,
        "forcefield": forcefield,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "net_charge_e": float(charges.sum()),
        "max_charge_correction_e": float(
            np.max(np.abs(charges - observed)) if charges.size else 0.0
        ),
    }
    return CovalentParameterBundle(molecule, system, charges, cache_key, provenance)
