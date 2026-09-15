from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import openmm as mm
import yaml
from openff.toolkit import Molecule
from openff.units import unit as offunit
from openmm import app, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from rdkit import Chem
from rdkit.Chem import AllChem

from atom_openmm.covalent_parameters import (
    CovalentParameterBundle,
    VirtualSiteParameter,
    _system_charges,
)


SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_ID = "gaff2-resp-cl-ep-v1"
SIGMA_HOLE_SMARTS = "[#6:1]-[#17X1:2]"


class LigandParameterizationError(RuntimeError):
    pass


def _write_yaml_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def _canonical_identity(molecule: Molecule) -> dict:
    return {
        "mapped_isomeric_smiles": molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=False
        ),
        "formal_charge": int(
            round(molecule.total_charge.m_as(offunit.elementary_charge))
        ),
    }


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_protocol(raw: dict | None) -> dict:
    raw = dict(raw or {})
    qm = dict(raw.get("qm") or {})
    sigma = dict(raw.get("sigma_holes") or {})
    protocol = {
        "id": str(raw.get("id", DEFAULT_PROTOCOL_ID)),
        "forcefield": str(raw.get("forcefield", "gaff-2.2.20")),
        "conformers": int(raw.get("conformers", 5)),
        "conformer_seed": int(raw.get("conformer_seed", 20260915)),
        "qm": {
            "engine": str(qm.get("engine", "gaussian16")),
            "geometry_method": str(qm.get("geometry_method", "HF")),
            "geometry_basis": str(qm.get("geometry_basis", "6-31G*")),
            "esp_method": str(qm.get("esp_method", "HF")),
            "esp_basis": str(qm.get("esp_basis", "6-31G*")),
            "cores_per_conformer": int(qm.get("cores_per_conformer", 7)),
            "memory_mb_per_conformer": int(
                qm.get("memory_mb_per_conformer", 12000)
            ),
            "parallel_conformers": int(qm.get("parallel_conformers", 5)),
            "executable": str(qm.get("executable", "g16")),
        },
        "sigma_holes": {
            "smarts": str(sigma.get("smarts", SIGMA_HOLE_SMARTS)),
            "distance_a": float(sigma.get("distance_a", 1.64)),
        },
        "resp": {
            "qwt": float((raw.get("resp") or {}).get("qwt", 0.0005)),
            "executable": str((raw.get("resp") or {}).get("executable", "resp")),
            "espgen_executable": str(
                (raw.get("resp") or {}).get("espgen_executable", "espgen")
            ),
        },
    }
    if protocol["conformers"] < 1:
        raise LigandParameterizationError("protocol.conformers must be positive")
    if protocol["qm"]["parallel_conformers"] < 1:
        raise LigandParameterizationError("parallel_conformers must be positive")
    if protocol["qm"]["cores_per_conformer"] < 1:
        raise LigandParameterizationError("cores_per_conformer must be positive")
    if protocol["qm"]["memory_mb_per_conformer"] < 100:
        raise LigandParameterizationError("memory_mb_per_conformer is too small")
    if protocol["qm"]["engine"] != "gaussian16":
        raise LigandParameterizationError("only qm.engine: gaussian16 is supported")
    if (
        protocol["qm"]["geometry_method"] != protocol["qm"]["esp_method"]
        or protocol["qm"]["geometry_basis"] != protocol["qm"]["esp_basis"]
    ):
        raise LigandParameterizationError(
            "the current Gaussian protocol requires identical geometry and ESP methods"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", protocol["id"]):
        raise LigandParameterizationError(
            "protocol.id may contain only letters, digits, '.', '_', and '-'"
        )
    if not protocol["forcefield"].startswith("gaff-"):
        raise LigandParameterizationError("RESP artifacts currently require a GAFF force field")
    if protocol["sigma_holes"]["distance_a"] <= 0.0:
        raise LigandParameterizationError("sigma-hole distance must be positive")
    return protocol


def cache_identity(molecule: Molecule, protocol: dict) -> tuple[str, str]:
    identity = _canonical_identity(molecule)
    molecule_key = _digest(identity)
    scientific_protocol = json.loads(json.dumps(protocol))
    for key in (
        "cores_per_conformer", "memory_mb_per_conformer", "parallel_conformers",
        "executable",
    ):
        scientific_protocol["qm"].pop(key, None)
    scientific_protocol["resp"].pop("executable", None)
    scientific_protocol["resp"].pop("espgen_executable", None)
    artifact_key = _digest({
        "schema_version": SCHEMA_VERSION,
        "molecule": identity,
        "protocol": scientific_protocol,
    })
    return molecule_key, artifact_key


def _find_sigma_holes(molecule: Molecule, protocol: dict) -> list[tuple[int, int, int]]:
    rdkit = molecule.to_rdkit()
    query = Chem.MolFromSmarts(protocol["sigma_holes"]["smarts"])
    if query is None or query.GetNumAtoms() != 2:
        raise LigandParameterizationError("sigma-hole SMARTS must contain exactly two atoms")
    matches = rdkit.GetSubstructMatches(query, uniquify=True, useChirality=True)
    sites = []
    for carbon, chlorine in matches:
        if rdkit.GetAtomWithIdx(chlorine).GetAtomicNum() != 17:
            raise LigandParameterizationError("sigma-hole SMARTS atom 2 must be chlorine")
        frame_candidates = sorted(
            atom.GetIdx()
            for atom in rdkit.GetAtomWithIdx(carbon).GetNeighbors()
            if atom.GetIdx() != chlorine
        )
        if not frame_candidates:
            raise LigandParameterizationError(
                f"chlorine parent carbon {carbon} has no virtual-site frame atom"
            )
        sites.append((carbon, chlorine, frame_candidates[0]))
    return sites


def _generate_conformers(molecule: Molecule, count: int, seed: int) -> list[Chem.Mol]:
    source = molecule.to_rdkit()
    if source.GetNumConformers() != 1:
        raise LigandParameterizationError("input ligand must contain one 3D conformer")
    outputs = []
    first = Chem.Mol(source)
    outputs.append(first)
    if count == 1:
        return outputs
    generated = Chem.Mol(source)
    generated.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.enforceChirality = True
    ids = list(AllChem.EmbedMultipleConfs(generated, numConfs=count - 1, params=params))
    if len(ids) != count - 1:
        raise LigandParameterizationError(
            f"RDKit generated {len(ids)} of {count - 1} requested conformers"
        )
    if AllChem.MMFFHasAllMoleculeParams(generated):
        AllChem.MMFFOptimizeMoleculeConfs(generated, maxIters=1000)
    else:
        AllChem.UFFOptimizeMoleculeConfs(generated, maxIters=1000)
    for conformer_id in ids:
        item = Chem.Mol(generated)
        conformer = Chem.Conformer(generated.GetConformer(conformer_id))
        item.RemoveAllConformers()
        item.AddConformer(conformer, assignId=True)
        outputs.append(item)
    return outputs


def _gaussian_input(molecule: Chem.Mol, protocol: dict, stem: str) -> str:
    qm = protocol["qm"]
    charge = Chem.GetFormalCharge(molecule)
    electron_count = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms()) - charge
    if electron_count % 2:
        raise LigandParameterizationError(
            "open-shell ligands are not supported by the Gaussian RESP protocol"
        )
    multiplicity = 1
    conformer = molecule.GetConformer()
    lines = [
        f"%chk={stem}.chk",
        f"%nprocshared={qm['cores_per_conformer']}",
        f"%mem={qm['memory_mb_per_conformer']}MB",
        (
            f"#P {qm['geometry_method']}/{qm['geometry_basis']} Opt SCF=Tight "
            "NoSymm Pop=MK IOp(6/33=2,6/42=6)"
        ),
        "",
        f"AToM-OpenMM RESP conformer {stem}",
        "",
        f"{charge} {multiplicity}",
    ]
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2s} {point.x: .10f} {point.y: .10f} {point.z: .10f}")
    lines.extend(["", ""])
    return "\n".join(lines)


def _run_checked(command: list[str], *, cwd: Path, stdin=None, stdout=None) -> None:
    try:
        subprocess.run(
            command, cwd=cwd, stdin=stdin, stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LigandParameterizationError(f"command failed: {' '.join(command)}") from exc


def _run_gaussian_task(task: tuple[int, Chem.Mol], workdir: Path, protocol: dict) -> Path:
    index, molecule = task
    directory = workdir / f"conformer_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"conformer_{index:02d}"
    input_path = directory / f"{stem}.gjf"
    output_path = directory / f"{stem}.log"
    input_path.write_text(_gaussian_input(molecule, protocol, stem))
    with input_path.open() as source, output_path.open("w") as output:
        _run_checked([protocol["qm"]["executable"]], cwd=directory, stdin=source, stdout=output)
    if "Normal termination of Gaussian" not in output_path.read_text(errors="replace"):
        raise LigandParameterizationError(f"Gaussian did not terminate normally for conformer {index}")
    esp_path = directory / f"{stem}.esp"
    _run_checked(
        [protocol["resp"]["espgen_executable"], "-i", output_path.name, "-o", esp_path.name],
        cwd=directory,
    )
    return esp_path


def _completed_gaussian_esp(directory: Path, index: int) -> Path | None:
    stem = f"conformer_{index:02d}"
    output = directory / f"{stem}.log"
    esp = directory / f"{stem}.esp"
    if (
        output.is_file()
        and esp.is_file()
        and "Normal termination of Gaussian" in output.read_text(errors="replace")
    ):
        try:
            _parse_amber_esp(esp)
        except (OSError, ValueError, LigandParameterizationError):
            return None
        return esp
    return None


def _checkpoint_gaussian_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        try:
            temporary.replace(destination)
        except FileExistsError:
            pass
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parse_amber_esp(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        raise LigandParameterizationError(f"empty ESP file: {path}")
    try:
        # Amber uses two adjacent I5 fields.  Large grids therefore produce
        # headers such as ``   4311275`` for 43 atoms and 11275 ESP points.
        natom = int(lines[0][:5])
        nesp = int(lines[0][5:10])
    except (TypeError, ValueError):
        raise LigandParameterizationError(f"invalid ESP header: {path}")
    if len(lines) != 1 + natom + nesp:
        raise LigandParameterizationError(f"ESP record count differs in {path}")
    atoms = np.asarray([[float(x) for x in line.split()[-3:]] for line in lines[1:1+natom]])
    records = np.asarray([[float(x) for x in line.split()[-4:]] for line in lines[1+natom:]])
    return atoms, records[:, 0], records[:, 1:]


def _site_coordinates(atom_coordinates_bohr: np.ndarray, sites, distance_a: float) -> np.ndarray:
    distance_bohr = float(distance_a) / 0.529177210903
    output = []
    for carbon, chlorine, _ in sites:
        axis = atom_coordinates_bohr[chlorine] - atom_coordinates_bohr[carbon]
        norm = np.linalg.norm(axis)
        if norm <= 1.0e-8:
            raise LigandParameterizationError("C-Cl distance is zero in ESP geometry")
        output.append(atom_coordinates_bohr[chlorine] + distance_bohr * axis / norm)
    return np.asarray(output)


def _write_multi_esp(path: Path, datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    with Path(path).open("w") as handle:
        for centers, potentials, points in datasets:
            handle.write(f"{len(centers):5d}{len(potentials):5d}\n")
            for xyz in centers:
                handle.write("".join(f"{value:16.7E}" for value in xyz) + "\n")
            for potential, xyz in zip(potentials, points):
                handle.write(f"{potential:16.7E}" + "".join(f"{value:16.7E}" for value in xyz) + "\n")


def _resp_input(
    atomic_numbers: list[int], site_count: int, conformer_count: int, formal_charge: int, qwt: float
) -> str:
    centers = list(atomic_numbers) + [0] * site_count
    lines = [
        "AToM-OpenMM multi-conformer RESP with sigma-hole sites",
        " &cntrl",
        f" nmol={conformer_count}, ihfree=1, iqopt=1, qwt={qwt:.8f},",
        " &end",
    ]
    for conformer in range(conformer_count):
        lines.extend([
            " 1.0",
            f"conformer {conformer + 1}",
            f"{formal_charge:5d}{len(centers):5d}",
        ])
        for element in centers:
            lines.append(f"{element:5d}{0:5d}")
        lines.append("")
    lines.extend(["", ""])
    if conformer_count > 1:
        for atom_index in range(1, len(centers) + 1):
            lines.append(f"{conformer_count:5d}")
            pairs = [
                value
                for conformer in range(1, conformer_count + 1)
                for value in (conformer, atom_index)
            ]
            lines.append("".join(f"{value:5d}" for value in pairs))
    lines.extend(["", ""])
    return "\n".join(lines)


def _parse_resp_charges(path: Path, center_count: int, conformer_count: int) -> np.ndarray:
    text = Path(path).read_text()
    if "*" in text:
        raise LigandParameterizationError(
            f"RESP charge output overflowed its fixed-width field: {path}"
        )
    tokens = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+)(?:[EeDd][-+]?\d+)?",
        text,
    )
    values = np.asarray([float(value.replace("D", "E")) for value in tokens])
    if values.size == center_count:
        return values
    if values.size == center_count * conformer_count:
        matrix = values.reshape(conformer_count, center_count)
        if not np.allclose(matrix, matrix[0], atol=5.0e-6):
            raise LigandParameterizationError("RESP equivalent conformers have different charges")
        return matrix[0]
    raise LigandParameterizationError(
        f"RESP returned {values.size} charges; expected {center_count} or {center_count * conformer_count}"
    )


def _relative_esp_error(centers, charges, potentials, points) -> float:
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    if np.any(distances <= 1.0e-10):
        raise LigandParameterizationError("ESP point coincides with a charge center")
    predicted = np.sum(charges[None, :] / distances, axis=1)
    denominator = float(np.sum(potentials * potentials))
    if denominator <= 0.0:
        raise LigandParameterizationError("ESP reference norm is zero")
    return float(np.sqrt(np.sum((predicted - potentials) ** 2) / denominator))


def _run_resp(workdir: Path, molecule: Molecule, sites, protocol, esp_paths) -> tuple[np.ndarray, dict]:
    raw = [_parse_amber_esp(path) for path in esp_paths]
    expected_atoms = molecule.n_atoms
    if any(len(atoms) != expected_atoms for atoms, _, _ in raw):
        raise LigandParameterizationError("Gaussian ESP atom count differs from the ligand")
    distance_a = protocol["sigma_holes"]["distance_a"]
    augmented = []
    atom_only = []
    for atoms, potentials, points in raw:
        site_xyz = _site_coordinates(atoms, sites, distance_a)
        augmented.append((np.vstack((atoms, site_xyz)), potentials, points))
        atom_only.append((atoms, potentials, points))
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    formal_charge = int(round(molecule.total_charge.m_as(offunit.elementary_charge)))

    def fit(label, datasets, site_count):
        directory = workdir / label
        directory.mkdir(parents=True, exist_ok=True)
        esp = directory / "esp.dat"
        inp = directory / "resp.in"
        out = directory / "resp.out"
        pch = directory / "resp.pch"
        charges = directory / "resp.chg"
        _write_multi_esp(esp, datasets)
        inp.write_text(_resp_input(
            atomic_numbers, site_count, len(datasets), formal_charge,
            protocol["resp"]["qwt"],
        ))
        _run_checked([
            protocol["resp"]["executable"], "-O", "-i", inp.name,
            "-o", out.name, "-p", pch.name, "-t", charges.name, "-e", esp.name,
        ], cwd=directory)
        return _parse_resp_charges(charges, molecule.n_atoms + site_count, len(datasets))

    fitted = fit("resp_sigma_hole" if sites else "resp_atom_only", augmented, len(sites))
    control = fit("resp_atom_only_control", atom_only, 0) if sites else fitted.copy()
    site_charges = fitted[molecule.n_atoms:]
    if not np.all(np.isfinite(fitted)) or np.any(site_charges <= 0.0):
        raise LigandParameterizationError("RESP sigma-hole charges must be positive and finite")
    total = float(fitted.sum())
    if not np.isclose(total, formal_charge, atol=5.0e-5):
        raise LigandParameterizationError(
            f"RESP total charge {total:.8f} differs from formal charge {formal_charge}"
        )
    ep_errors = [
        _relative_esp_error(centers, fitted, potentials, points)
        for centers, potentials, points in augmented
    ]
    control_errors = [
        _relative_esp_error(centers, control, potentials, points)
        for centers, potentials, points in atom_only
    ]
    if sites and float(np.mean(ep_errors)) >= float(np.mean(control_errors)):
        raise LigandParameterizationError("sigma-hole RESP fit did not improve aggregate ESP error")
    return fitted, {
        "relative_rmse_per_conformer": ep_errors,
        "atom_only_relative_rmse_per_conformer": control_errors,
        "relative_rmse_mean": float(np.mean(ep_errors)),
        "atom_only_relative_rmse_mean": float(np.mean(control_errors)),
    }


def _gaff_system(
    molecule: Molecule,
    charges_e: np.ndarray,
    forcefield_name: str,
    virtual_sites: tuple[VirtualSiteParameter, ...],
) -> mm.System:
    parameterized = Molecule(molecule)
    typing_charges = np.asarray(charges_e, dtype=float).copy()
    for site in virtual_sites:
        typing_charges[int(site.parent_atom_indices[1])] += float(site.charge_e)
    parameterized.partial_charges = typing_charges * offunit.elementary_charge
    generator = GAFFTemplateGenerator(molecules=[parameterized], forcefield=forcefield_name)
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(generator.generator)
    system = forcefield.createSystem(
        parameterized.to_topology().to_openmm(), nonbondedMethod=app.NoCutoff,
        constraints=None, rigidWater=False, removeCMMotion=False,
    )
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, mm.NonbondedForce)
    )
    for index, charge in enumerate(charges_e):
        _, sigma, epsilon = nonbonded.getParticleParameters(index)
        nonbonded.setParticleParameters(
            index, float(charge) * unit.elementary_charge, sigma, epsilon
        )
    observed = _system_charges(system)
    if not np.allclose(observed, charges_e, atol=1.0e-6):
        raise LigandParameterizationError("GAFF system did not preserve RESP charges")
    return system


def _write_mol2_and_frcmod(workdir: Path, source: Path, charges: np.ndarray) -> tuple[Path, Path]:
    charges_path = workdir / "atomic_charges.txt"
    charges_path.write_text("\n".join(f"{value:.10f}" for value in charges) + "\n")
    mol2 = workdir / "ligand.mol2"
    frcmod = workdir / "ligand.frcmod"
    suffix = source.suffix.lower().lstrip(".")
    _run_checked([
        "antechamber", "-i", str(source), "-fi", suffix, "-o", str(mol2),
        "-fo", "mol2", "-at", "gaff2", "-c", "rc", "-cf", str(charges_path),
        "-pf", "y", "-s", "0",
    ], cwd=workdir)
    _run_checked([
        "parmchk2", "-i", str(mol2), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2",
    ], cwd=workdir)
    return mol2, frcmod


def _publish_artifact(config_path: Path, config: dict) -> Path:
    ligand_path = Path(config["ligand"])
    if not ligand_path.is_absolute():
        ligand_path = (config_path.parent / ligand_path).resolve()
    if not ligand_path.is_file():
        raise LigandParameterizationError(f"ligand does not exist: {ligand_path}")
    cache_dir = Path(config["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (config_path.parent / cache_dir).resolve()
    protocol = normalize_protocol(config.get("protocol"))
    molecule = Molecule.from_file(str(ligand_path), allow_undefined_stereo=False)
    molecule.generate_unique_atom_names()
    molecule_key, artifact_key = cache_identity(molecule, protocol)
    artifact = cache_dir / "artifacts" / artifact_key
    manifest_path = artifact / "manifest.yaml"
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        if manifest.get("status") == "completed":
            return artifact
    cache_dir.mkdir(parents=True, exist_ok=True)
    configured_workdir = os.environ.get("ATOM_PARAMETERIZATION_WORKDIR")
    work_parent = Path(
        configured_workdir or os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    work_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"atom-param-{artifact_key[:12]}-", dir=work_parent))
    try:
        sites = _find_sigma_holes(molecule, protocol)
        conformers = _generate_conformers(
            molecule, protocol["conformers"], protocol["conformer_seed"]
        )
        qm_root = staging / "qm"
        tasks = list(enumerate(conformers, start=1))
        esp_paths = [None] * len(tasks)
        progress_root = cache_dir / "work" / artifact_key / "qm"
        missing_tasks = []
        for task in tasks:
            index = task[0]
            checkpoint = progress_root / f"conformer_{index:02d}"
            checkpoint_esp = _completed_gaussian_esp(checkpoint, index)
            if checkpoint_esp is None:
                missing_tasks.append(task)
                continue
            local = qm_root / checkpoint.name
            shutil.copytree(checkpoint, local)
            esp_paths[index - 1] = local / checkpoint_esp.name
        workers = min(protocol["qm"]["parallel_conformers"], len(tasks))
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(_run_gaussian_task, task, qm_root, protocol): task[0]
                for task in missing_tasks
            }
            for future in as_completed(pending):
                index = pending[future]
                try:
                    esp_paths[index - 1] = future.result()
                    _checkpoint_gaussian_directory(
                        qm_root / f"conformer_{index:02d}",
                        progress_root / f"conformer_{index:02d}",
                    )
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise failures[0]
        if any(path is None for path in esp_paths):
            raise LigandParameterizationError("one or more Gaussian ESP outputs are missing")
        fitted, diagnostics = _run_resp(staging, molecule, sites, protocol, esp_paths)
        atom_charges = fitted[:molecule.n_atoms]
        virtual_sites = tuple(
            VirtualSiteParameter(
                name=f"CL_EP_{index + 1}", kind="sigma_hole",
                parent_atom_indices=tuple(int(value) for value in parents),
                distance_a=protocol["sigma_holes"]["distance_a"],
                charge_e=float(fitted[molecule.n_atoms + index]),
            )
            for index, parents in enumerate(sites)
        )
        system = _gaff_system(
            molecule, atom_charges, protocol["forcefield"], virtual_sites
        )
        molecule.partial_charges = atom_charges * offunit.elementary_charge
        molecule.to_file(str(staging / "ligand.sdf"), file_format="SDF")
        (staging / "system.xml").write_text(mm.XmlSerializer.serialize(system))
        portable_charges = atom_charges.copy()
        for site in virtual_sites:
            portable_charges[int(site.parent_atom_indices[1])] += float(site.charge_e)
        _write_mol2_and_frcmod(staging, ligand_path, portable_charges)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "artifact_key": artifact_key,
            "molecule_key": molecule_key,
            "molecule": _canonical_identity(molecule),
            "protocol": protocol,
            "source": str(ligand_path),
            "atom_count": molecule.n_atoms,
            "atomic_charges_e": atom_charges.tolist(),
            "mol2_atomic_charges_e": portable_charges.tolist(),
            "virtual_sites": [asdict(site) for site in virtual_sites],
            "total_charge_e": float(fitted.sum()),
            "esp_diagnostics": diagnostics,
            "files": {
                "molecule": "ligand.sdf", "system": "system.xml",
                "mol2": "ligand.mol2", "frcmod": "ligand.frcmod", "qm": "qm",
            },
        }
        _write_yaml_atomic(staging / "manifest.yaml", manifest)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        publish = Path(
            tempfile.mkdtemp(prefix=f".{artifact_key}.", dir=artifact.parent)
        )
        try:
            shutil.copytree(staging, publish, dirs_exist_ok=True)
            try:
                publish.replace(artifact)
            except FileExistsError:
                winner = yaml.safe_load((artifact / "manifest.yaml").read_text()) or {}
                if winner.get("status") != "completed":
                    raise LigandParameterizationError(
                        f"concurrent cache artifact is incomplete: {artifact}"
                    )
        finally:
            if publish.exists():
                shutil.rmtree(publish)
        index_path = cache_dir / "index" / molecule_key / f"{protocol['id']}.yaml"
        _write_yaml_atomic(index_path, {
            "schema_version": SCHEMA_VERSION,
            "artifact_key": artifact_key,
            "protocol_id": protocol["id"],
        })
        progress_artifact = cache_dir / "work" / artifact_key
        if progress_artifact.exists():
            shutil.rmtree(progress_artifact)
        return artifact
    except Exception:
        failure_root = (
            cache_dir / "work" / artifact_key /
            f"postprocessing_failure_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        )
        failure_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "resp_sigma_hole", "resp_atom_only", "resp_atom_only_control",
        ):
            source = staging / name
            destination = failure_root / name
            if source.is_dir() and not destination.exists():
                shutil.copytree(source, destination)
        raise
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _isomorphic_atom_map(stored: Molecule, requested: Molecule) -> dict[int, int]:
    query = stored.to_rdkit()
    target = requested.to_rdkit()
    if query.GetNumAtoms() != target.GetNumAtoms():
        raise LigandParameterizationError("cached ligand atom count differs")
    match = target.GetSubstructMatch(query, useChirality=True)
    reverse = query.GetSubstructMatch(target, useChirality=True)
    if len(match) != query.GetNumAtoms() or len(reverse) != target.GetNumAtoms():
        raise LigandParameterizationError("cached ligand is not isomorphic to the requested ligand")
    return {stored_index: int(requested_index) for stored_index, requested_index in enumerate(match)}


def _reorder_system(system: mm.System, old_to_new: dict[int, int]) -> mm.System:
    count = system.getNumParticles()
    if set(old_to_new) != set(range(count)) or set(old_to_new.values()) != set(range(count)):
        raise LigandParameterizationError("cached system atom map is incomplete")
    output = mm.System()
    inverse = {new: old for old, new in old_to_new.items()}
    for new in range(count):
        output.addParticle(system.getParticleMass(inverse[new]))
    for index in range(system.getNumConstraints()):
        a, b, distance = system.getConstraintParameters(index)
        output.addConstraint(old_to_new[int(a)], old_to_new[int(b)], distance)
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            copied = mm.HarmonicBondForce()
            for index in range(force.getNumBonds()):
                a, b, length, k = force.getBondParameters(index)
                copied.addBond(old_to_new[int(a)], old_to_new[int(b)], length, k)
        elif isinstance(force, mm.HarmonicAngleForce):
            copied = mm.HarmonicAngleForce()
            for index in range(force.getNumAngles()):
                a, b, c, angle, k = force.getAngleParameters(index)
                copied.addAngle(old_to_new[int(a)], old_to_new[int(b)], old_to_new[int(c)], angle, k)
        elif isinstance(force, mm.PeriodicTorsionForce):
            copied = mm.PeriodicTorsionForce()
            for index in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, k = force.getTorsionParameters(index)
                copied.addTorsion(*(old_to_new[int(x)] for x in (a, b, c, d)), periodicity, phase, k)
        elif isinstance(force, mm.NonbondedForce):
            copied = mm.NonbondedForce()
            copied.setNonbondedMethod(force.getNonbondedMethod())
            copied.setCutoffDistance(force.getCutoffDistance())
            for new in range(count):
                copied.addParticle(*force.getParticleParameters(inverse[new]))
            for index in range(force.getNumExceptions()):
                a, b, charge, sigma, epsilon = force.getExceptionParameters(index)
                copied.addException(old_to_new[int(a)], old_to_new[int(b)], charge, sigma, epsilon)
        else:
            raise LigandParameterizationError(
                f"cached ligand system contains unsupported force {type(force).__name__}"
            )
        copied.setName(force.getName())
        output.addForce(copied)
    return output


def load_cached_parameters(
    ligand: Path, *, cache_dir: Path, protocol_id: str = DEFAULT_PROTOCOL_ID
) -> CovalentParameterBundle:
    ligand = Path(ligand).resolve()
    requested = Molecule.from_file(str(ligand), allow_undefined_stereo=False)
    requested.generate_unique_atom_names()
    identity = _canonical_identity(requested)
    molecule_key = _digest(identity)
    index_path = Path(cache_dir) / "index" / molecule_key / f"{protocol_id}.yaml"
    if not index_path.is_file():
        raise LigandParameterizationError(
            f"no cached {protocol_id} parameters for {ligand.name}: {index_path}"
        )
    index = yaml.safe_load(index_path.read_text()) or {}
    artifact = Path(cache_dir) / "artifacts" / str(index.get("artifact_key", ""))
    manifest_path = artifact / "manifest.yaml"
    if not manifest_path.is_file():
        raise LigandParameterizationError(f"cached artifact is incomplete: {artifact}")
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "completed":
        raise LigandParameterizationError(f"cached artifact manifest is invalid: {manifest_path}")
    manifest_protocol = manifest.get("protocol") or {}
    if (
        manifest.get("artifact_key") != index.get("artifact_key")
        or manifest.get("molecule") != identity
        or manifest_protocol.get("id") != protocol_id
    ):
        raise LigandParameterizationError("cached artifact identity or protocol differs")
    stored = Molecule.from_file(str(artifact / manifest["files"]["molecule"]), allow_undefined_stereo=False)
    atom_map = _isomorphic_atom_map(stored, requested)
    stored_system = mm.XmlSerializer.deserialize((artifact / manifest["files"]["system"]).read_text())
    system = _reorder_system(stored_system, atom_map)
    stored_charges = np.asarray(manifest["atomic_charges_e"], dtype=float)
    if stored_charges.shape != (requested.n_atoms,):
        raise LigandParameterizationError("cached atomic charge count differs from the ligand")
    charges = np.empty_like(stored_charges)
    for old, new in atom_map.items():
        charges[new] = stored_charges[old]
    requested.partial_charges = charges * offunit.elementary_charge
    sites = []
    for raw in manifest.get("virtual_sites", []):
        parents = tuple(atom_map[int(index)] for index in raw["parent_atom_indices"])
        sites.append(VirtualSiteParameter(
            name=str(raw["name"]), kind=str(raw["kind"]),
            parent_atom_indices=parents, distance_a=float(raw["distance_a"]),
            charge_e=float(raw["charge_e"]), sigma_a=float(raw.get("sigma_a", 0.0)),
            epsilon_kj_mol=float(raw.get("epsilon_kj_mol", 0.0)),
        ))
    total_charge = float(charges.sum()) + sum(site.charge_e for site in sites)
    formal_charge = float(requested.total_charge.m_as(offunit.elementary_charge))
    if not np.isclose(total_charge, formal_charge, atol=5.0e-5):
        raise LigandParameterizationError(
            "cached atomic and virtual-site charges do not sum to the formal charge"
        )
    return CovalentParameterBundle(
        molecule=requested, system=system, charges_e=charges,
        cache_key=str(manifest["artifact_key"]),
        provenance={
            "charge_model": "resp-sigma-hole", "ligand_forcefield": manifest_protocol["forcefield"],
            "parameter_artifact": str(artifact), "artifact_key": manifest["artifact_key"],
            "protocol_id": protocol_id, "esp_diagnostics": manifest.get("esp_diagnostics", {}),
        },
        virtual_sites=tuple(sites),
    )


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text()) or {}
    if payload.get("schema_version", 1) != SCHEMA_VERSION:
        raise LigandParameterizationError("unsupported parameterization schema_version")
    if not isinstance(payload.get("ligand"), str) or not payload["ligand"].strip():
        raise LigandParameterizationError("ligand must be a non-empty path")
    if not isinstance(payload.get("cache_dir"), str) or not payload["cache_dir"].strip():
        raise LigandParameterizationError("cache_dir must be a non-empty path")
    normalize_protocol(payload.get("protocol"))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate cached GAFF2/RESP ligand parameters")
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-key", action="store_true", help="print the cache key without running QM")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.print_key:
        ligand = Path(config["ligand"])
        if not ligand.is_absolute():
            ligand = (config_path.parent / ligand).resolve()
        molecule = Molecule.from_file(str(ligand), allow_undefined_stereo=False)
        _, key = cache_identity(molecule, normalize_protocol(config.get("protocol")))
        print(key)
        return 0
    artifact = _publish_artifact(config_path, config)
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
