import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../code/')
import fit_perturber as fp

import itertools
import pathlib
from schwimmbad.utils import batch_tasks
import yaml
import time
import numpy as np
import astropy.table as at

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic


import astropy.coordinates as coord
import astropy.units as u
from pyia import GaiaData

def worker(batch):
    (batch_id, _) , tasks, cache_path = batch
    cache_file = cache_path / f'cached_lik_{batch_id:04d}.fits'

    if cache_file.exists():
        return cache_file

    tbl = at.Table(names=('b', 'psi', 'z', 'v_z', 'v_psi', 't', 'logm', 'core',
                          'll_model', 'll_model_short', 'll_phi2_short', 'll_data',
                          'gap_ratio', 'pert_apo', 'pert_peri'))

    PreModel = fp.FitPert(data, data_short, mw, w0_now)
    for params in tasks:
        params = np.around(params, decimals=1)
        try:
            ll, ll_short, ll_phi2_short, ll_data, gap_ratio, apo, peri = PreModel.loglik(params)
        except:
            ll, ll_short, ll_phi2_short, ll_data, gap_ratio, apo, peri = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        rows = np.append(params,
                         np.around([ll, ll_short, ll_phi2_short, ll_data, gap_ratio, apo, peri],
                                   decimals=2))
        #print(rows) #loglikelihoods, density ratio
        tbl.add_row(rows)
    tbl.write(cache_file, overwrite=True)

    return cache_file

def init():

        #########################
        ## DATA FOR COMPARISON ##
        #########################
        after = GaiaData('../data/member_prob_all.fits')
        model_output = after[after.post_member_prob > 0.25]
        data = model_output[(model_output.phi1[:,0] > -65) & (model_output.phi1[:,0] < -22)]
        data_short = model_output[(model_output.phi1[:,0] > -43) & (model_output.phi1[:,0] < -25)]


        ##########################################
        ## CURRENT STREAM WITHOUT THE PERTURBER ##
        ##########################################
        # all this is result of orbit fitting with phi1 fixed
        df = ms.FardalStreamDF(random_state=np.random.RandomState(42))
        gd1_init = gc.GD1Koposov10(phi1 = -13*u.degree, phi2=0*u.degree, distance=8.836*u.kpc,
                              pm_phi1_cosphi2=-10.575*u.mas/u.yr,
                              pm_phi2=-2.439*u.mas/u.yr,
                             radial_velocity = -189.6*u.km/u.s)
        rep = gd1_init.transform_to(coord.Galactocentric).data
        gd1_w0 = gd.PhaseSpacePosition(rep)
        gd1_mass = 5e3 * u.Msun
        gd1_pot = gp.PlummerPotential(m=gd1_mass, b=5*u.pc, units=galactic)
        mw = gp.MilkyWayPotential(halo={'m': 5.43e11*u.Msun, 'r_s': 15.78*u.kpc})
        gen_gd1 = ms.MockStreamGenerator(df, mw, progenitor_potential=gd1_pot)
        gd1_stream, gd1_nbody = gen_gd1.run(gd1_w0, gd1_mass,
                                        dt=-1 * u.Myr, n_steps=3000)
        gd1 = gd1_stream.to_coord_frame(gc.GD1)

        gd1_short = gd1_stream[(-65<gd1.phi1.value) & (gd1.phi1.value<-22)]
        w0_now = gd.PhaseSpacePosition(gd1_short.data, gd1_short.vel)

        return data, data_short, mw, w0_now

def main(pool, config_file):

    print('Creating grid...')
    config_file = pathlib.Path(config_file)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    b_grid = config['grid_b']
    #psi_grid = np.arange(config['min_psi'], config['max_psi'], config['grid_psi'])
    psi_grid = config['grid_psi']
    z_grid = np.arange(config['min_z'], config['max_z'], config['grid_z'])
    vz_grid = np.arange(config['min_vz'], config['max_vz'], config['grid_vz'])
    vpsi_grid = np.arange(config['min_vpsi'], config['max_vpsi'], config['grid_vpsi'])
    t_grid = np.arange(config['min_t'], config['max_t'], config['grid_t'])
    logm_grid = np.arange(config['min_logm'], config['max_logm'], config['grid_logm'])
    core_grid = config['grid_core']

    grid = {'b_grid': b_grid,
            'psi_grid': psi_grid,
            'z_grid': z_grid,
            'vz_grid': vz_grid,
            'vpsi_grid': vpsi_grid,
            't_grid': t_grid,
            'logm_grid': logm_grid,
            'core_grid': core_grid}

    combinations = []
    for values in itertools.product(*grid.values()):
        combinations.append(np.array(values))

    print(len(combinations))

    cache_path = config_file.parent / 'cache'
    cache_path.mkdir(exist_ok=True)

    # Create batched tasks to send out to MPI workers
    batched_tasks = batch_tasks(n_batches=max(pool.size * 5 - 1, 1),
                        arr=combinations,
                        args=(cache_path, ))

    filenames = []
    for filename in pool.map(worker, batched_tasks):
        filenames.append(filename)


data, data_short, mw, w0_now = init()

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    # Define parser object
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    parser.add_argument("-c", "--config", dest="config_file", required=True, type=str)

    args = parser.parse_args()


    # deal with multiproc:
    if args.mpi:
        from schwimmbad.mpi import MPIPool
        Pool = MPIPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()

    with Pool(**kw) as pool:
        main(pool=pool, config_file = args.config_file)

    sys.exit(0)
