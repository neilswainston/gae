'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=wrong-import-order
from rdkit import Chem

from gae import train
import pandas as pd


def _load_data(filename):
    '''Load data.'''

    df = pd.read_csv(filename)

    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        print(mol)

    return adj, features


def main(args):
    '''main method.'''

    # Load data:
    adj, features = _load_data(args[0])

    # Train:
    train.train(adj, features)


if __name__ == '__main__':
    main(sys.argv[1:])
