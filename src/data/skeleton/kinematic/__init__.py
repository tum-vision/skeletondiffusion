from .utils import get_adj_matrix
from .freeman import FreeManKinematic
from .h36m import H36MKinematic
from .amass import AMASSKinematic

def get_kinematic_objclass(dataset_name):
    dataset_type = {'h36m': 'H36M', 'freeman': 'FreeMan', 'amass': 'AMASS', 'amass-mano': 'AMASS', 
                    '3dpw': 'AMASS' }[dataset_name.lower()]
    return globals()[dataset_type+"Kinematic"], dataset_type