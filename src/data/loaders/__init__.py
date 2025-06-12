from .freeman  import  FreeManDataset
from .h36m  import  H36MDataset
from .amass  import  AMASSDataset
from .amass_zeroshot import ZeroShotAMASSDataset
from .base import custom_collate_for_mmgt

class D3PWZeroShotDataset(ZeroShotAMASSDataset):
    dataset_name = '3dpw'