import itertools
import yaml
import numpy as np

config_file = 'config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
b = config['grid_b']
#psi = np.arange(config['min_psi'], config['max_psi'], config['grid_psi'])
psi = config['grid_psi']
z = np.arange(config['min_z'], config['max_z'], config['grid_z'])
vz = np.arange(config['min_vz'], config['max_vz'], config['grid_vz'])
vpsi = np.arange(config['min_vpsi'], config['max_vpsi'], config['grid_vpsi'])
t = np.arange(config['min_t'], config['max_t'], config['grid_t'])
logm = np.arange(config['min_logm'], config['max_logm'], config['grid_logm'])
logcore = config['grid_core']

ncombinations = len(b)*len(psi)*len(z)*len(vz)*len(vpsi)*len(t)*len(logm)*len(logcore)

print('Total number of combinations is: {}'.format(ncombinations))
total_hours = ncombinations * 5 / (128*20 *3600)
print('Assuming 5 seconds per calculation on 20 cores with 128 nodes each, this will take about {} hours'.format(total_hours))
