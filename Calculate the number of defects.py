from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
import numpy as np
import ovito

#dump file is the file which is after irradiation,the path can be written in absolute path
#In ovito, the relative path is ..\ovito\
node = import_file("dump.irradiation",multiple_frames=True)

# Perform Wigner-Seitz analysis:
ws = WignerSeitzAnalysisModifier(
    per_type_occupancies = True, 
    eliminate_cell_deformation = True)
ws.reference.load("dump.start")  #this dump file is the file which after equilibration and before irradiation
node.modifiers.append(ws)

# Define a modifier function that selects N-vacancy
def modify(frame, input, output):

    # Retrieve the two-dimensional Numpy array with the site occupancy numbers.
    occupancies = input.particle_properties['Occupancy'].array
    
    # Get the site types as additional input:
    site_type = input.particle_properties.particle_type.array

    # Calculate total occupancy of every site:
    total_occupancy = np.sum(occupancies, axis=1)

    #In our system, Particle type 1 is A atom, Particle type 2 is B atom
    #site_type means the particle type in the original lattice  
    #the value of occupancie[:,0] means the number of A atom(s) in the present lattice
    #the value of occupancie[:,1] means the number of B atom(s) in the present lattice
    #You can also use "total_occupancy=XX" to achieve your goal
    selection1 = (site_type == 1) & (occupancies[:,0] == 0) & (occupancies[:,1]==0)   #Vacancy A
    selection2 = (site_type == 2) & (occupancies[:,0] == 0) & (occupancies[:,1]==0)   #Vacancy B
    selection3 = (site_type == 1) & (occupancies[:,0] == 1) & (occupancies[:,1]==1)   #Interstitial B
    selection4 = (site_type == 2) & (occupancies[:,0] == 1) & (occupancies[:,1]==1)   #Interstitial A
    selection5 = (site_type == 1) & (occupancies[:,0] == 2) & (occupancies[:,1]==0)   #Interstitial A
    selection6 = (site_type == 2) & (occupancies[:,0] == 0) & (occupancies[:,1]==2)   #Interstitial B
    selection7 = (site_type == 1) & (occupancies[:,0] == 0) & (occupancies[:,1]==1)   #Antisite B
    selection8 = (site_type == 2) & (occupancies[:,0] == 1) & (occupancies[:,1]==0)   #Antisite A
    
    # Additionally output the total number of antisites as a global attribute:(As follows, it only shows some examples)
    output.attributes['Ga_vacancy_count'] = np.count_nonzero(selection1)
    output.attributes['N_vacancy_count'] = np.count_nonzero(selection2)
    output.attributes['vacancy_count'] = np.count_nonzero(selection1+selection2)
    
# Insert Python modifier into the data pipeline.
node.modifiers.append(PythonScriptModifier(function = modify))

# Let OVITO do the computation and export the number of identified 
# antisites as a function of simulation time to a text file:
export_file(node, "defects.txt", "txt", 
    columns = ['Timestep', 'Ga_vacancy_count','N_vacancy_count','vacancy_count'],
    multiple_frames = True)