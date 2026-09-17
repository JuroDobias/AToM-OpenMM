#! python

# Usage: python make_atm_rbfe_system_frompdb.py <options>
# Emilio Gallicchio, 5/2023 adapted from code by Bill Swope, 11/2021

############################################
#                                          #
#   IMPORTS                                #
#                                          #
############################################

import os, sys
import string
import numpy as np
from datetime import datetime
from time import time

# following for argument passing tools
import argparse

# OpenMM components
from openmm import XmlSerializer
import openmm as mm
from openmm import Vec3
from openmm.app import PDBFile
from openmm.app import ForceField, Modeller
from openmm.app import PME, HBonds, NoCutoff

# OpenFF components from the toolkit
from openff.toolkit.topology import Molecule
from openff.units import unit as offunit

from openmm.unit import angstrom, nanometer, amu, molar

# OpenFF and OpenMM components for ligand force field parameters
from openff.toolkit.topology import Molecule


def _infer_solvent_model(solventforcefield):
    solvent_files = " ".join(str(ff).lower() for ff in solventforcefield)
    if "opc" in solvent_files:
        return "tip4pew"
    if "tip5p" in solvent_files:
        return "tip5p"
    if "tip4p" in solvent_files:
        return "tip4pew"
    if "spce" in solvent_files or "spc/e" in solvent_files:
        return "spce"
    return "tip3p"


def boundingBoxSizes(positions):
    xmin = positions[0][0]
    xmax = positions[0][0]
    ymin = positions[0][1]
    ymax = positions[0][1]
    zmin = positions[0][2]
    zmax = positions[0][2]
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]
        z = positions[i][2]
        # print('type of x ', type(x))
        # print('Site ', i, ' Coord ', positions[i])
        if(x > xmax):
            xmax = x
        if(x < xmin):
            xmin = x
        if(y > ymax):
            ymax = y
        if(y < ymin):
            ymin = y
        if(z > zmax):
            zmax = z
        if(z < zmin):
            zmin = z
    return [ (xmin,xmax), (ymin,ymax), (zmin,zmax) ] 

def assign_chain_ids(topology):
    """
    Assigns 'A', 'B', 'C', etc., to chains in an OpenMM topology 
    that currently have no ID assigned.
    """
    # Create a generator for letters A-Z, then AA, AB, etc. if needed
    # string.ascii_uppercase provides 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    potential_names = list(string.ascii_uppercase)
    
    name_index = 0
    
    for chain in topology.chains():
        # Check if ID is None, empty, or just whitespace
        if not chain.id or chain.id.strip() == "":
            if name_index < len(potential_names):
                new_id = potential_names[name_index]
                chain.id = new_id
                print(f"Assigned ID '{new_id}' to chain {chain.index}")
                name_index += 1
            else:
                print(f"Warning: Ran out of unique letters for chain {chain.index}")
        else:
            print(f"Chain {chain.index} already has ID: '{chain.id}'")


def _cached_typing_molecule(parameters):
    molecule = Molecule(parameters.molecule)
    charges = np.asarray(parameters.charges_e, dtype=float).copy()
    for site in parameters.virtual_sites:
        charges[int(site.parent_atom_indices[1])] += float(site.charge_e)
    molecule.partial_charges = charges * offunit.elementary_charge
    return molecule


def _nonbonded_force(system):
    forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(forces) != 1:
        raise ValueError(f"cached ATM setup requires one NonbondedForce; found {len(forces)}")
    return forces[0]


def _exception_map(force):
    output = {}
    for index in range(force.getNumExceptions()):
        values = force.getExceptionParameters(index)
        output[tuple(sorted((int(values[0]), int(values[1]))))] = values[2:]
    return output


def _coulomb_14_scale(force):
    scales = []
    for index in range(force.getNumExceptions()):
        atom1, atom2, charge_product, _, epsilon = force.getExceptionParameters(index)
        q1 = force.getParticleParameters(int(atom1))[0]
        q2 = force.getParticleParameters(int(atom2))[0]
        denominator = (q1 * q2).value_in_unit(mm.unit.elementary_charge**2)
        product = charge_product.value_in_unit(mm.unit.elementary_charge**2)
        if abs(denominator) > 1.0e-10 and abs(product) > 1.0e-10:
            scales.append(product / denominator)
        elif epsilon.value_in_unit(mm.unit.kilojoules_per_mole) > 0.0:
            scales.append(1.0 / 1.2)
    return float(np.median(scales)) if scales else 1.0 / 1.2


def _install_ligand_parameters(system, topology, positions, parameters, residue_name):
    atoms = [atom for atom in topology.atoms() if atom.residue.name == residue_name]
    if len(atoms) != parameters.molecule.n_atoms:
        raise ValueError(
            f"parameterized {residue_name} atom count {parameters.molecule.n_atoms} differs from "
            f"prepared residue count {len(atoms)}"
        )
    force = _nonbonded_force(system)
    cached = _nonbonded_force(parameters.system)
    for source_index, atom in enumerate(atoms):
        _, sigma, epsilon = force.getParticleParameters(atom.index)
        _, cached_sigma, cached_epsilon = cached.getParticleParameters(source_index)
        if not np.isclose(
            sigma.value_in_unit(mm.unit.nanometer),
            cached_sigma.value_in_unit(mm.unit.nanometer),
            atol=1.0e-8,
        ) or not np.isclose(
            epsilon.value_in_unit(mm.unit.kilojoules_per_mole),
            cached_epsilon.value_in_unit(mm.unit.kilojoules_per_mole),
            atol=1.0e-8,
        ):
            raise ValueError(
                f"parameterized {residue_name} Lennard-Jones parameters differ at atom {source_index}"
            )
        force.setParticleParameters(
            atom.index,
            float(parameters.charges_e[source_index]) * mm.unit.elementary_charge,
            sigma,
            epsilon,
        )

    system_atom_indices = {atom.index for atom in atoms}
    for exception_index in range(force.getNumExceptions()):
        atom1, atom2, _, sigma, epsilon = force.getExceptionParameters(exception_index)
        if int(atom1) not in system_atom_indices or int(atom2) not in system_atom_indices:
            continue
        q1 = force.getParticleParameters(int(atom1))[0]
        q2 = force.getParticleParameters(int(atom2))[0]
        if epsilon.value_in_unit(mm.unit.kilojoules_per_mole) > 0.0:
            charge_product = q1 * q2 / 1.2
        else:
            charge_product = 0.0 * mm.unit.elementary_charge**2
        force.setExceptionParameters(
            exception_index, atom1, atom2, charge_product, sigma, epsilon
        )

    exception_parameters = _exception_map(force)
    scale14 = _coulomb_14_scale(force)
    chain = topology.addChain("X")
    extra_residue_name = "E1" if residue_name == "L1" else "E2"
    residue = topology.addResidue(extra_residue_name, chain)
    values = list(positions.value_in_unit(mm.unit.nanometer))
    site_indices = []
    source_to_system = [atom.index for atom in atoms]
    for site in parameters.virtual_sites:
        if site.kind != "sigma_hole" or len(site.parent_atom_indices) != 3:
            raise ValueError(f"unsupported ATM virtual site {site.kind!r}")
        carbon, halogen, frame = (
            source_to_system[int(index)] for index in site.parent_atom_indices
        )
        exception_parameters = _exception_map(force)
        particle = system.addParticle(0.0 * mm.unit.dalton)
        system.setVirtualSite(
            particle,
            mm.LocalCoordinatesSite(
                [halogen, carbon, frame],
                [1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, -1.0, 1.0],
                mm.Vec3(float(site.distance_a) / 10.0, 0.0, 0.0),
            ),
        )
        force.addParticle(
            float(site.charge_e) * mm.unit.elementary_charge,
            max(float(site.sigma_a), 0.01) * mm.unit.angstrom,
            float(site.epsilon_kj_mol) * mm.unit.kilojoules_per_mole,
        )
        halogen_charge = force.getParticleParameters(halogen)[0]
        for other in range(particle):
            if other == halogen:
                force.addException(
                    particle, other, 0.0 * mm.unit.elementary_charge**2,
                    1.0 * mm.unit.nanometer, 0.0 * mm.unit.kilojoules_per_mole,
                )
                continue
            source = exception_parameters.get(tuple(sorted((halogen, other))))
            if source is None:
                continue
            charge_product, sigma, _ = source
            other_charge = force.getParticleParameters(other)[0]
            denominator = (halogen_charge * other_charge).value_in_unit(
                mm.unit.elementary_charge**2
            )
            source_product = charge_product.value_in_unit(mm.unit.elementary_charge**2)
            if abs(source_product) <= 1.0e-12:
                scale = 0.0
            elif abs(denominator) > 1.0e-12:
                scale = source_product / denominator
            else:
                scale = scale14
            force.addException(
                particle,
                other,
                float(site.charge_e) * mm.unit.elementary_charge * other_charge * scale,
                sigma,
                0.0 * mm.unit.kilojoules_per_mole,
            )
        topology.addAtom(site.name, None, residue)
        halogen_xyz = np.asarray(values[halogen], dtype=float)
        carbon_xyz = np.asarray(values[carbon], dtype=float)
        direction = halogen_xyz - carbon_xyz
        direction /= np.linalg.norm(direction)
        values.append(halogen_xyz + direction * float(site.distance_a) / 10.0)
        site_indices.append(particle)
    return values * mm.unit.nanometer, site_indices


# Backward-compatible name for callers that install cached RESP bundles.
_install_cached_ligand_parameters = _install_ligand_parameters

# Example Usage:
# from openmm.app import PDBFile
# pdb = PDBFile('input.pdb')
# assign_chain_ids(pdb.topology)

def make_system(
        receptorfile,
        displacement,
        xmloutfile,
        pdboutfile,
        lig1file=None,
        lig2file=None,
        lig1sdffile=None,
        lig2sdffile=None,
        cofsdffile=None,
        proteinforcefield=['amber14-all.xml'],
        solventforcefield=['amber14/tip3p.xml'],
        ligandforcefield='openff-2.0.0',
        ffcachefile=None,
        implsolv=None,
        hmass=1.0,
        ionicstrength=0.15,
        solvent_model=None,
        template_generator_kwargs=None,
        ligandchargemodel=None,
        ligandparametercache=None,
        ligandparameterprotocol=None,
        ligandsigmaholes=None,
        flagverbose=False
    ):
    print('Generate ATM RBFE OpenMM System')
    today = datetime.today()
    print('\nDate and time at start: ', today.strftime('%c'))
    program_start_timer = time()
    
    if lig1sdffile is not None:
        print('Warning: LIG1SDFinFile id deprecated. Use LIG1inFile')
        if lig1file is None:
            lig1file = lig1sdffile

    if lig2sdffile is not None:
        print('Warning: LIG2SDFinFile id deprecated. Use LIG2inFile')
        if lig2file is None:
            lig2file = lig2sdffile
            
    #catch abfe or rbfe
    rbfe = False
    if lig2file  is not None:
        rbfe = True

    if isinstance(displacement, str):
        displacement = [float(r) for r in displacement.split()]
    displacement = Vec3(*displacement) * angstrom

    #implicit solvent
    if implsolv == 'None':
        implsolv = None

    hmass = float(hmass)


    #####################################################
    #   Echo out the suppliable input parameters        #
    #####################################################


    print('\nUser-supplied input parameters')
    print('Receptor file name:                 ', receptorfile)
    print('Protein force field:                ', proteinforcefield)
    print('Solvent/ion force field             ', solventforcefield )
    print('Solvent packing model:              ', solvent_model)
    print('Ligand force field:                 ', ligandforcefield)
    print('Template generator kwargs:          ', template_generator_kwargs)
    print('Ligand 1 file name:                 ', lig1file)
    if rbfe:
        print('Ligand 2 file name:                 ', lig2file)
        print('Displacement                        ', displacement)
    print('Topology PDB output file:           ', pdboutfile)
    print('System XML output file:             ', xmloutfile)
    print('Force field cache file:             ', ffcachefile)


    print('Call ForceField for protein and water')
    forcefield = ForceField(*proteinforcefield,*solventforcefield)
    if solvent_model is None:
        solvent_model = _infer_solvent_model(solventforcefield)
    print('Using solvent packing model:        ', solvent_model)
    if implsolv is not None:
        if implsolv == "OBC2":
            forcefield.loadFile('implicit/obc2.xml')
        elif implsolv == "GBN2":
            forcefield.loadFile('implicit/gbn2.xml')
        elif implsolv == "HCT":
            forcefield.loadFile('implicit/hct.xml')
        elif implsolv == "Vacuum" or implsolv == "vacuum":
            pass
        else:
            print('Unknown implicit solvent %s' % implsolv)
            sys.exit(1)

    ligand_parameters = None
    if ligandchargemodel == "resp-sigma-hole" or ligandsigmaholes is not None:
        if not rbfe:
            raise ValueError("sigma-hole setup currently requires ATM RBFE")
        from atom_openmm.hybrid_parameters import parameterize_ligand
        parameter_options = {
            "ligand_forcefield": ligandforcefield,
            "ligand_charge_model": ligandchargemodel,
            "ligand_parameter_cache": ligandparametercache,
            "ligand_parameter_protocol": ligandparameterprotocol,
            "ligand_sigma_holes": ligandsigmaholes,
        }
        ligand_parameters = (
            parameterize_ligand(
                lig1file,
                **parameter_options,
            ),
            parameterize_ligand(
                lig2file,
                **parameter_options,
            ),
        )

    # to store OpenFF molecule objects of non-protein units
    ligandmolecules = []

    ############################################
    #                                          #
    #   READ AND CHARACTERIZE RECEPTOR         #
    #                                          #
    ############################################

    rcptpext = os.path.splitext(receptorfile)[1]
    rcpt_ommtopology = None
    rcpt_positions = None
    if rcptpext == '.pdb':
        print('Receptor in PDB format')
        pdbrcpt = PDBFile(receptorfile)
        rcpt_positions = pdbrcpt.positions
        rcpt_ommtopology = pdbrcpt.topology
        from atom_openmm.hybrid_systems import _repair_template_bonds
        _repair_template_bonds(rcpt_ommtopology, rcpt_positions, forcefield)
    elif rcptpext == '.sdf':
        print('Receptor in SDF format')
        molrcpt = Molecule.from_file(receptorfile, file_format='SDF',
                                    allow_undefined_stereo=True)
        ligandmolecules.append(molrcpt)

        pos = molrcpt.conformers[0].to('angstrom').magnitude
        rcpt_positions = [Vec3(pos[i][0], pos[i][1], pos[i][2]) for i in range(pos.shape[0])] * angstrom
        
        offtopology = molrcpt.to_topology()
        rcpt_ommtopology = offtopology.to_openmm(ensure_unique_atom_names=True)
    else:
        print("Error: Unrecognized receptor file name: %s" % receptorfile)
        sys.exit(1)

    nrcpt = rcpt_ommtopology.getNumAtoms()
    print('Number of atoms in receptor:', nrcpt)

    print('Call Modeller: include receptor')
    assign_chain_ids(rcpt_ommtopology)
    modeller = Modeller(rcpt_ommtopology, rcpt_positions)

    print("Calculating receptor bounding box:")
    bbox = boundingBoxSizes(rcpt_positions)
    bboxsizes = [ bbox[i][1]-bbox[i][0] for i in range(3) ]
    bboxfaces = [ bboxsizes[2]*bboxsizes[1], bboxsizes[2]*bboxsizes[0],  bboxsizes[1]*bboxsizes[0] ]
    print("Areas of faces", bboxfaces)
    smallest_direction = 0
    smallest_area = bboxfaces[0]
    for i in range(3):
        if bboxfaces[i] < smallest_area:
            smallest_direction = i
    print("Direction of smallest area dimension:", smallest_direction)

        
    ############################################
    #                                          #
    #   READ AND CHARACTERIZE LIGANDS          #
    #                                          #
    ############################################

    if cofsdffile is not None:
        print('Read cofactor:')
        molcof = Molecule.from_file(cofsdffile, file_format='SDF',
                                allow_undefined_stereo=True)
        ligandmolecules.append(molcof)
        molcof_ommtopology = molcof.to_topology().to_openmm(ensure_unique_atom_names=True)

        #assign the residue name, assumes one residue
        resfile = os.path.split(cofsdffile)[1]
        resname = os.path.splitext(resfile)[0]
        for residue in molcof_ommtopology.residues():
            residue.name = resname.upper()

        pos = molcof.conformers[0].to('angstrom').magnitude
        molcof_positions = [Vec3(pos[i][0], pos[i][1], pos[i][2]) for i in range(pos.shape[0])] * angstrom
        ncof = molcof_ommtopology.getNumAtoms()
        print('Number of atoms in cofactor:', ncof)
        print('Call Modeller: include cofactor')
        modeller.add(molcof_ommtopology, molcof_positions)


    print('Read ligand 1 from %s:' % lig1file)
    
    fileext = (os.path.splitext(lig1file)[1]).upper()
    if fileext in ('.SDF', '.MOL2'):
        file_format = 'SDF' if fileext == '.SDF' else 'MOL2'
        mollig1 = (
            _cached_typing_molecule(ligand_parameters[0])
            if ligand_parameters is not None
            else Molecule.from_file(lig1file, file_format=file_format, allow_undefined_stereo=True)
        )
        ligandmolecules.append(mollig1)
        lig1_ommtopology = mollig1.to_topology().to_openmm(ensure_unique_atom_names=True)
        pos = mollig1.conformers[0].to('angstrom').magnitude
        lig1_positions = [Vec3(pos[i][0], pos[i][1], pos[i][2]) for i in range(pos.shape[0])] * angstrom
        #assign the residue name, assumes one residue
        resname_lig1 = "L1"
        for residue in lig1_ommtopology.residues():
            residue.name = resname_lig1
        chainname_lig1 = "L"
        for chain in lig1_ommtopology.chains():
            chain.id = chainname_lig1
    elif fileext == '.PDB':
        lig1pdb = PDBFile(lig1file)
        lig1_ommtopology = lig1pdb.topology
        lig1_positions = lig1pdb.positions
        chainname_lig1 = "L"
        for chain in lig1_ommtopology.chains():
            chain.id = chainname_lig1
    else:
        print("Error: unrecognized file: %s" % lig1file)
        sys.exit(1)

    nlig1 = lig1_ommtopology.getNumAtoms()
    print('Number of atoms in ligand 1:', nlig1)
    print('Call Modeller: include ligand 1')
    modeller.add(lig1_ommtopology, lig1_positions)

    if not rbfe:
        # if ABFE, translate the ligand 1 coordinates into the solvent to calculate the
        # bounding box below
        for i in range(nlig1):
            lig1_positions[i] += displacement
    else:
        # RBFE mode:
        # read ligand 2 and place it in the solvent
        print('Read ligand 2 from %s:' % lig2file)
        fileext = (os.path.splitext(lig2file)[1]).upper()
        if fileext in ('.SDF', '.MOL2'):
            file_format = 'SDF' if fileext == '.SDF' else 'MOL2'
            mollig2 = (
                _cached_typing_molecule(ligand_parameters[1])
                if ligand_parameters is not None
                else Molecule.from_file(lig2file, file_format=file_format, allow_undefined_stereo=True)
            )
            ligandmolecules.append(mollig2)
            lig2_ommtopology = mollig2.to_topology().to_openmm(ensure_unique_atom_names=True)
            pos = mollig2.conformers[0].to('angstrom').magnitude
            lig2_positions = [Vec3(pos[i][0], pos[i][1], pos[i][2]) for i in range(pos.shape[0])] * angstrom
            #assign the residue name, assumes one residue
            resname_lig2 = "L2"
            for residue in lig2_ommtopology.residues():
                residue.name = resname_lig2
            chainname_lig2 = "M"
            for chain in lig2_ommtopology.chains():
                chain.id = chainname_lig2
        elif fileext == '.PDB':
            lig2pdb = PDBFile(lig2file)
            lig2_ommtopology = lig2pdb.topology
            lig2_positions = lig2pdb.positions
            chainname_lig2 = "M"
            for chain in lig2_ommtopology.chains():
                chain.id = chainname_lig2
        else:
            print("Error: unrecognized file: %s" % lig2file)
            sys.exit(1)

        nlig2 = lig2_ommtopology.getNumAtoms()
        print('Number of atoms in ligand 2:', nlig2)
        for i in range(nlig2):
            lig2_positions[i] += displacement
        print('Call Modeller: include ligand 2')
        modeller.add(lig2_ommtopology, lig2_positions)
        lig2atom_indexes = [ i for i in range(nrcpt+nlig1,nrcpt+nlig1+nlig2)]

    print("Calculating system bounding box:")
    if not rbfe:
        bbox = boundingBoxSizes(rcpt_positions + lig1_positions)
    else:
        bbox = boundingBoxSizes(rcpt_positions + lig1_positions + lig2_positions)
    bboxsizes = [ bbox[i][1]-bbox[i][0] for i in range(3) ]
    padding = 2. * 1.0*nanometer
    xBoxvec = Vec3((bboxsizes[0]+padding)/nanometer, 0., 0.)*nanometer
    yBoxvec = Vec3(0.0, (bboxsizes[1]+padding)/nanometer, 0.)*nanometer
    zBoxvec = Vec3(0.0, 0.0, (bboxsizes[2]+padding)/nanometer)*nanometer
    print("boxVectors:", (xBoxvec,yBoxvec,zBoxvec ))



    #bboxfaces = [ bboxsizes[2]*bboxsizes[1], bboxsizes[2]*bboxsizes[0],  bboxsizes[1]*bboxsizes[0] ]
    #print("Areas of faces", bboxfaces)
    #smallest_direction = 0
    #smallest_area = bboxfaces[0]
    #for i in range(3):
    #    if bboxfaces[i] < smallest_area:
    #        smallest_direction = i
    #print("Smallest direction", smallest_direction)


        
    ############################################
    #                                          #
    #   SET UP FORCEFIELD FOR LIGANDS          #
    #                                          #
    ############################################

    print('\nSet up the combined protein + ligand + water system for simulation')
    template_gen = None
    if ligandforcefield[0:4] == "gaff":
        from openmmforcefields.generators import GAFFTemplateGenerator
        print('Using GAFFTemplateGenerator function for ligands')
        template_gen = GAFFTemplateGenerator(
            molecules=ligandmolecules,
            forcefield=ligandforcefield,
            cache=ffcachefile,
            template_generator_kwargs=template_generator_kwargs,
        )
    elif ligandforcefield[0:6] == "openff":
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        print('Call SMIRNOFFTemplateGenerator function for ligands')
        template_gen = SMIRNOFFTemplateGenerator(molecules=ligandmolecules, forcefield=ligandforcefield, cache=ffcachefile, template_generator_kwargs=template_generator_kwargs )
    elif ligandforcefield[0:8] == "espaloma":
        from openmmforcefields.generators import EspalomaTemplateGenerator
        print('Call EspalomaTemplateGenerator function for ligands')
        template_gen = EspalomaTemplateGenerator(molecules=ligandmolecules, forcefield=ligandforcefield, cache=ffcachefile, template_generator_kwargs=template_generator_kwargs )
    else:
        print('Unknown ligand force field %s' % ligandforcefield)
        sys.exit(1)

    # Register the SMIRNOFF template generator
    # NOTE: forcefield object was initialized (above)
    # for the protein + water (subsystem) force field 
    # This step adds support for the ligand force field
    forcefield.registerTemplateGenerator(template_gen.generator)

    if implsolv is None:
        print("Ionic strength = ", ionicstrength*molar)
        print("Adding solvent and processing system ...")
        modeller.addExtraParticles(forcefield)
        modeller.addSolvent(forcefield, model=solvent_model, boxVectors = (xBoxvec,yBoxvec,zBoxvec ), ionicStrength = ionicstrength*molar)
        modeller.addExtraParticles(forcefield)
        print("Number of atoms in solvated system:", modeller.topology.getNumAtoms())
        system=forcefield.createSystem(modeller.topology, nonbondedMethod = PME, nonbondedCutoff = 0.9*nanometer,
                                    constraints=HBonds, rigidWater = True, removeCMMotion = False, hydrogenMass = hmass*amu)
    else:
        print("Solvent model: %s" % implsolv)
        print("Number of atoms in implicit solvent system:", modeller.topology.getNumAtoms())
        print("Processing system ...")
        system=forcefield.createSystem(modeller.topology, nonbondedMethod = NoCutoff,
                                    constraints=HBonds, rigidWater = True, removeCMMotion = False, hydrogenMass = hmass*amu)

    output_positions = modeller.positions
    if ligand_parameters is not None:
        output_positions, _ = _install_ligand_parameters(
            system, modeller.topology, output_positions, ligand_parameters[0], "L1"
        )
        output_positions, _ = _install_ligand_parameters(
            system, modeller.topology, output_positions, ligand_parameters[1], "L2"
        )
        if modeller.topology.getNumAtoms() != system.getNumParticles():
            raise ValueError("parameterized ATM topology and System particle counts differ")

    with open(xmloutfile, 'w') as output:
        output.write(XmlSerializer.serialize(system))

    if pdboutfile is not None:
        PDBFile.writeFile(modeller.topology, output_positions,
                        open(pdboutfile,'w'), keepIds=True)

    today = datetime.today()
    print('\n\nDate and time at end:   ', today)
    program_end_timer = time()
    print('\nTotal compute time %.3f seconds' %  (program_end_timer-program_start_timer))


def main():
    whatItDoes = """
    Produces an .xml file with the OpenMM's system for ATM relative binding
    free energy calcuations.

    The user supplies a pdb file that contains protein, cofactors, ligands, 
    water, ions, probably prepared by a different software package 
    (Maestro, OpenEye).  Cofactors and ligands have to be described in an
    sdf file.  Then force fields are assigned to all components

    Emilio Gallicchio 5/2023
    adapted from simProteinLigandWater.py script by Bill Swope 11/2021
    """

    parser = argparse.ArgumentParser(description=whatItDoes)

    # Required input
    parser.add_argument('--receptorinFile', required=True,  type=str, default=None, dest='receptorfile',
                        help='Receptor in SDF (.sdf) or PDB format (.pdb)')
    parser.add_argument('--displacement',  required=True,  type=str, default=None, dest='displacement',
                        help='string with displacement vector in angstroms like "22. 0.0 0.0" ')
    parser.add_argument('--systemXMLoutFile',  required=True,  type=str, default=None, dest='xmloutfile',
                        help='Name of the XML file where to save the System')
    parser.add_argument('--systemPDBoutFile', required=True, type=str, default=None, dest='pdboutfile',
                        help='Name of the PDB file where to output to system')

    # Optional input
    parser.add_argument('--LIG1inFile',  required=False,  type=str, default=None, dest='lig1file',
                        help='First ligand in SDF (.sdf) or PDB format (.pdb)')
    parser.add_argument('--LIG2inFile',  required=False,  type=str, default=None, dest='lig2file',
                        help='Second ligand in SDF (.sdf) or PDB format (.pdb)')
    parser.add_argument('--LIG1SDFinFile',  required=False,  type=str, default=None, dest='lig1sdffile',
                        help='SDF file of first ligand')
    parser.add_argument('--LIG2SDFinFile',  required=False,  type=str, default=None, dest='lig2sdffile',
                        help='SDF file of second ligand')
    parser.add_argument('--cofactorsSDFFile',  required=False,  type=str, default=None, dest='cofsdffile',
                        help='SDF file with receptor cofactors')

    parser.add_argument('--proteinForceField', required=False, action="append", dest='proteinforcefield',
                        help='Force field for protein: default amber14-all.xml ', default=['amber14-all.xml'])
    parser.add_argument('--solventForceField', required=False, action="append", dest='solventforcefield',
                        help='Force field for solvent/ions: default amber14/tip3p.xml ', default=['amber14/tip3p.xml'])
    parser.add_argument('--ligandForceField', required=False, type=str, dest='ligandforcefield',
                        default='openff-2.0.0',
                        help='Force field for ligand:  openff-2.0.0 (default), gaff, or espaloma-0.3.2')
    parser.add_argument('--implicitSolvent', required=False, type=str, dest='implsolv',
                        default=None,
                        help='Implicit solvent to use: HCT OBC2 GBn2, vacuum.')
    parser.add_argument('--forcefieldJSONCachefile', required=False, type=str, dest='ffcachefile',
                        default=None,
                        help='Force field ligand cache database')
    parser.add_argument('--hmass', required=False, type=float, dest='hmass',
                        default=1.0,
                        help='Hydrogen mass, set it to 1.5 amu to use a 4 fs time-step')
    parser.add_argument('--ionicStrength', required=False, type=float, dest='ionicstrength',
                        default=0.15,
                        help='Total concentration of monoatomic ions to add')
    
    # Arguments that are flags
    parser.add_argument('--verbose', required=False, action='store_true', dest='flagverbose',
                        help='Get more output with this flag')

    args = vars(parser.parse_args())
    make_system(**args)


if __name__ == "__main__":
    main()
