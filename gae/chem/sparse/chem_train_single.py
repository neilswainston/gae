'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=wrong-import-order
from rdkit import Chem
import scipy

from gae.tf import train_single
import numpy as np
import pandas as pd


def _load_data(filename):
    '''Load data.'''
    df = pd.read_csv(filename)
    smiles = df['smiles'][0]
    adj, features = _get_data(smiles)
    return adj, features


def _get_data(smiles):
    '''Get data from SMILES.'''
    mol = Chem.MolFromSmiles(smiles)

    adj = scipy.sparse.lil_matrix(
        (mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=int)

    for bond in mol.GetBonds():
        adj[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1

    features = np.array([[atom.GetAtomicNum(),
                          atom.GetMass(),
                          atom.GetExplicitValence(),
                          atom.GetFormalCharge()]
                         for atom in mol.GetAtoms()])

    return scipy.sparse.csr_matrix(adj), scipy.sparse.lil_matrix(features)


def main():
    '''main method.'''

    # Load data:
    filename = 'data/spectra.csv'
    adj, features = _load_data(filename)

    # Train:
    train_single.train(adj, features, epochs=10000)


if __name__ == '__main__':
    main()
