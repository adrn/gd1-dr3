import numpy as np
from astropy.table import Table, vstack
import glob

if __name__ == '__main__':
    files = glob.glob('cache/*')

    base_table = Table(names=('b', 'psi', 'z', 'v_z', 'v_psi', 't', 'logm',
                              'll_model', 'll_model_short', 'll_phi2_short', 'll_data', 'gap_ratio', 'pert_apo', 'pert_peri'))
    #for file in files:
    #    table = Table.read(file)
    #    #tables = np.append(tables, table)
    #    base_table = vstack([base_table, table])
    base_table = vstack([Table.read(file) for file in files])

    all_table = base_table[np.flip(base_table.argsort('ll_model'))]
    all_table.write('full_table0.fits', format='fits', overwrite=True)
