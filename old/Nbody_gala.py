## THIS CODE IS TAKEN DIRECTLY FROM GALA BUT I HAVE TAKEN OUT SOME PARTS (UNIT VERIFICATION) THAT ARE NOT NEEDED FOR THIS PROJECT TO IMPROVE THE SPEED OF COMPUTATION

import numpy as np

from gala.potential import Hamiltonian, NullPotential, StaticFrame
from gala.units import UnitSystem
from gala.util import atleast_2d
from gala.integrate.timespec import parse_time_specification
from gala.dynamics import Orbit, PhaseSpacePosition

from gala.dynamics.nbody.nbody import direct_nbody_dop853

__all__ = ['DirectNBody']

class DirectNBody:

    def __init__(self, w0, particle_potentials, external_potential=None,
                 frame=None, units=None, save_all=True):
        """Perform orbit integration using direct N-body forces between
        particles, optionally in an external background potential.
        TODO: could add another option, like in other contexts, for
        "extra_force" to support, e.g., dynamical friction
        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`
            The particle initial conditions.
        partcle_potentials : list
            List of potential objects to add mass or mass distributions to the
            particles. Use ``None`` to treat particles as test particles.
        external_potential : `~gala.potential.PotentialBase` subclass instance (optional)
            The background or external potential to integrate the particle
            orbits in.
        frame : :class:`~gala.potential.frame.FrameBase` subclass (optional)
            The reference frame to perform integratiosn in.
        units : `~gala.units.UnitSystem` (optional)
            Set of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        save_all : bool (optional)
            Save the full orbits of each particle. If ``False``, only returns
            the final phase-space positions of each particle.
        """
        units = particle_potentials[0].units
        frame = StaticFrame(units)

        self.units = units
        self.external_potential = external_potential
        self.frame = frame
        self.particle_potentials = particle_potentials
        self.save_all = save_all

        self.H = Hamiltonian(self.external_potential,
                             frame=self.frame)
        if not self.H.c_enabled:
            raise ValueError("Input potential must be C-enabled: one or more "
                             "components in the input external potential are "
                             "Python-only.")

        self.w0 = w0
        
    @property
    def w0(self):
        return self._w0

    @w0.setter
    def w0(self, value):
        self._w0 = value
        self._cache_w0()


    def _cache_w0(self):
        # cache the position and velocity / prepare the initial conditions
        self._pos = atleast_2d(self.w0.xyz.decompose(self.units).value,
                               insert_axis=1)
        self._vel = atleast_2d(self.w0.v_xyz.decompose(self.units).value,
                               insert_axis=1)
        self._c_w0 = np.ascontiguousarray(np.vstack((self._pos, self._vel)).T)

    def integrate_orbit(self, **time_spec):
        """
        Integrate the initial conditions in the combined external potential
        plus N-body forces.
        This integration uses the `~gala.integrate.DOPRI853Integrator`.
        Parameters
        ----------
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.
        Returns
        -------
        orbit : `~gala.dynamics.Orbit`
            The orbits of the particles.
        """

        # Prepare the time-stepping array
        t = parse_time_specification(self.units, **time_spec)

        ws = direct_nbody_dop853(self._c_w0, t, self.H,
                                 self.particle_potentials,
                                 save_all=self.save_all)

        if self.save_all:
            pos = np.rollaxis(np.array(ws[..., :3]), axis=2)
            vel = np.rollaxis(np.array(ws[..., 3:]), axis=2)

            orbits = Orbit(
                pos=pos * self.units['length'],
                vel=vel * self.units['length'] / self.units['time'],
                t=t * self.units['time'],
                hamiltonian=self.H)

        else:
            pos = np.array(ws[..., :3]).T
            vel = np.array(ws[..., 3:]).T

            orbits = PhaseSpacePosition(
                pos=pos * self.units['length'],
                vel=vel * self.units['length'] / self.units['time'],
                frame=self.frame)

        return orbits
