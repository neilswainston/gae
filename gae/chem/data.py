'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=wrong-import-order
from rdkit import Chem

import numpy as np


def encode(smiles, max_num_atoms=256):
    '''Encode.'''
    mol = Chem.MolFromSmiles(smiles)

    num_atoms = mol.GetNumAtoms()

    if num_atoms > max_num_atoms:
        raise ValueError('%s contains too many atoms (%i)' %
                         (smiles, num_atoms))

    atoms = np.zeros(shape=(max_num_atoms), dtype=int)

    for idx in range(num_atoms):
        atoms[idx] = mol.GetAtomWithIdx(idx).GetAtomicNum()

    bonds = np.zeros(shape=(max_num_atoms, max_num_atoms), dtype=int)

    for bnd in mol.GetBonds():
        bonds[bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()] = bnd.GetBondType()

    return atoms, bonds


def decode(atoms, bonds):
    '''Decode.'''
    mol = Chem.RWMol()

    for atomic_num in atoms:
        if atomic_num:
            atom = Chem.Atom(int(atomic_num))
            mol.AddAtom(atom)

    for begin_idx, bnds in enumerate(bonds):
        for end_idx, bond_type in enumerate(bnds):
            if bond_type:
                mol.AddBond(begin_idx, end_idx,
                            Chem.BondType.values[bond_type])

    return Chem.MolToSmiles(mol)


def main():
    '''main method.'''
    smiles = 'CCO'
    atoms, bonds = encode(smiles)
    decoded = decode(atoms, bonds)
    print(decoded)


if __name__ == '__main__':
    main()
