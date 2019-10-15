from ase.atoms import Atoms
from ase.quaternions import Quaternions
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.utils import basestring
from collections import deque
from ase import units ## ssrokyz
import numpy as np ## ssrokyz

def read_lammps_dump(fileobj, index, order=True, atomsobj=Atoms):
    """Method which reads a LAMMPS dump file.

    index: must be slice object
    order: Order the particles according to their id. Might be faster to
    switch it off.
    """
    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    def add_quantity(fields, var, labels, atom_attributes):
        for label in labels:
            if label not in atom_attributes:
                return
        var.append([float(fields[atom_attributes[label]])
                    for label in labels])
        
    def reorder(inlist, id):
        if not len(inlist):
            return inlist
        outlist = [None] * len(id)
        for i, v in zip(id, inlist):
            outlist[i - 1] = v
        return outlist

    def read_a_loop(f):
        ## Time step
        line = f.readline()
        assert 'ITEM: TIMESTEP' in line
        lo = []
        hi = []
        tilt = []
        id = []
        types = []
        positions = []
        element = [] ## ssrokyz
        scaled_positions = []
        velocities = []
        forces = []
        quaternions = []
        # Read out timestep
        line = f.readline()

        ## Number of atoms
        line = f.readline()
        assert 'ITEM: NUMBER OF ATOMS' in line
        line = f.readline()
        natoms = int(line.split()[0])
            
        ## Box bounds
        line = f.readline()
        assert 'ITEM: BOX BOUNDS' in line
        # save labels behind "ITEM: BOX BOUNDS" in
        # triclinic case (>=lammps-7Jul09)
        tilt_items = line.split()[3:]
        for i in range(3):
            line = f.readline()
            fields = line.split()
            lo.append(float(fields[0]))
            hi.append(float(fields[1]))
            if (len(fields) >= 3):
                tilt.append(float(fields[2]))

        # determine cell tilt (triclinic case!)
        if (len(tilt) >= 3):
            # for >=lammps-7Jul09 use labels behind
            # "ITEM: BOX BOUNDS" to assign tilt (vector) elements ...
            if (len(tilt_items) >= 3):
                xy = tilt[tilt_items.index('xy')]
                xz = tilt[tilt_items.index('xz')]
                yz = tilt[tilt_items.index('yz')]
            # ... otherwise assume default order in 3rd column
            # (if the latter was present)
            else:
                xy = tilt[0]
                xz = tilt[1]
                yz = tilt[2]
        else:
            xy = xz = yz = 0
        xhilo = (hi[0] - lo[0]) - (xy**2)**0.5 - (xz**2)**0.5
        yhilo = (hi[1] - lo[1]) - (yz**2)**0.5
        zhilo = (hi[2] - lo[2])
        if xy < 0:
            if xz < 0:
                celldispx = lo[0] - xy - xz
            else:
                celldispx = lo[0] - xy
        else:
            celldispx = lo[0]
        celldispy = lo[1]
        celldispz = lo[2]

        cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]
        celldisp = [[celldispx, celldispy, celldispz]]
                
        line = f.readline()
        assert 'ITEM: ATOMS' in line
        # (reliably) identify values by labels behind
        # "ITEM: ATOMS" - requires >=lammps-7Jul09
        # create corresponding index dictionary before
        # iterating over atoms to (hopefully) speed up lookups...
        atom_attributes = {}
        for (i, x) in enumerate(line.split()[2:]):
            atom_attributes[x] = i
        for n in range(natoms):
            line = f.readline()
            fields = line.split()
            id.append(int(fields[atom_attributes['id']]))
            types.append(int(fields[atom_attributes['type']]))
            element.append(str(fields[atom_attributes['element']])) ## ssrokyz
            add_quantity(fields, positions, ['x', 'y', 'z'], atom_attributes)
            add_quantity(fields, scaled_positions, ['xs', 'ys', 'zs'], atom_attributes)
            add_quantity(fields, velocities, ['vx', 'vy', 'vz'], atom_attributes)
            add_quantity(fields, forces, ['fx', 'fy', 'fz'], atom_attributes)
            add_quantity(fields, quaternions, ['c_q[1]', 'c_q[2]',
                                               'c_q[3]', 'c_q[4]'], atom_attributes)

        if order:
            types = reorder(types, id)
            element = reorder(element, id)
            positions = reorder(positions, id)
            scaled_positions = reorder(scaled_positions, id)
            velocities = reorder(np.array(velocities) /units.fs /1e3, id) ## ssrokyz ## lammps metal unit: Ang./picosec ## units.fs *1e3 = units.ps
            forces = reorder(forces, id)
            quaternions = reorder(quaternions, id)

        ## Make 'Atoms' object
        if len(quaternions):
            atoms = Quaternions(
                symbols=element,
                positions=positions,
                cell=cell,
                celldisp=celldisp,
                quaternions=quaternions,
                )
        elif len(positions):
            atoms = atomsobj(
                symbols=element,
                positions=positions,
                celldisp=celldisp,
                cell=cell,
                )
        elif len(scaled_positions):
            atoms = atomsobj(
                symbols=element,
                scaled_positions=scaled_positions,
                celldisp=celldisp,
                cell=cell,
                )
        if len(velocities):
            atoms.set_velocities(velocities)
        if len(forces):
            calculator = SinglePointCalculator(atoms, energy=0.0, forces=forces)
            atoms.set_calculator(calculator)
        return atoms

    ## Main
    images = []
    # Get number of atoms
    while True:
        tmp_line = f.readline()
        if 'ITEM: NUMBER OF ATOMS' in tmp_line:
            natoms = int(f.readline())
            f.seek(0)
            break
    # Get index
    get_nline_bool = False
    if index.stop == None or index.stop >= 0:
        if index.start == None or index.start >= 0:
            index = slice(*index.indices(int(1e8)))
        elif index.start < 0:
            get_nline_bool = True
    elif index.stop < 0:
        get_nline_bool = True
    if get_nline_bool:
        from ss_util import get_number_of_lines
        nimages = int(get_number_of_lines(f) / (natoms+9))
        # Set stop of slice
        index = slice(*index.indices(nimages))

    ## Main
    for img_ind in range(index.stop):
        if img_ind >= index.start and (img_ind - index.start) % index.step == 0:
            try: yield read_a_loop(f)
            except KeyboardInterrupt: raise ValueError('Keyboard interrupted.')
            except:
                print('%%%% Warning !! File is not complete. Be aware. %%%%')
                print('%%%% Warning !! File is not complete. Be aware. %%%%')
                print('%%%% Warning !! File is not complete. Be aware. %%%%')
                print('%%%% Warning !! File is not complete. Be aware. %%%%')
                break
            else: pass
        else:
            for j in range(natoms + 9):
                line = f.readline()
                if not line: break
        if img_ind == index.stop:
            break
