import os
import numpy as np
import torch
from functools import partial
import pickle
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from .loaders import D3PWZeroShotDataset
from .loaders.base.create_dataset_utils import compute_stats, compute_multimodal_gt
from .skeleton import get_skeleton_class    

OUTPUT_3D = 'data_3d_3dpw'    
    
def preprocess_dataset(dataset_folder,  smpl_body_models_path, output_path=OUTPUT_3D):
    if os.path.exists(f"{output_path}.npz"):
        print('The dataset already exists at', f"{output_path}.npz")
        raise Exception(f'The dataset already exists at {output_path}')
        
    print('Converting original 3DPW dataset from', dataset_folder)
    image_path = os.path.join(dataset_folder, "imageFiles")
    dataset_path = os.path.join(dataset_folder, "sequenceFiles")

    comp_device =  'cuda'
    print("Using device:", comp_device)
    
    # initialize all needed resources
    genders = "male", "female"
    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    bm, faces = {}, {}
    for gender in genders:
        bm_fname = os.path.join(smpl_body_models_path, 'smplh/{}/model.npz'.format(gender))
        dmpl_fname = os.path.join(smpl_body_models_path, 'dmpls/{}/model.npz'.format(gender))
        
        bm[gender] = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)

        faces[gender] = c2c(bm[gender].f)

    print("All resources initialized")
    
    output = {}
    count = [0]*3
    nframes = [0]*3
    splits = os.listdir(dataset_path)
    for s, split in enumerate(os.listdir(dataset_path)):
        split_path = os.path.join(dataset_path, split)
        split = 'valid' if split == 'validation' else split
        output[split] = {}
        for pkl in os.listdir(split_path):
            with open(os.path.join(split_path, pkl), 'rb') as reader:
                annotations = pickle.load(reader, encoding='latin1')
            seq_name = os.path.splitext(pkl)[0]
            for actor_index in range(len(annotations['genders'])):

                # joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
                bdata = {k: annotations[k][actor_index]for k in list(annotations.keys()) if k in ['poses_60Hz', 'trans_60Hz', 'genders', 'betas', 'trans']}
                gender = 'male' if bdata['genders'] =='m' else 'female'

                time_length = len(bdata['trans_60Hz'])
                body_parms = {
                    'root_orient': torch.Tensor(bdata['poses_60Hz'][:, :3]).to(comp_device), # controls the global root orientation
                    'pose_body': torch.Tensor(bdata['poses_60Hz'][:, 3:66]).to(comp_device), # controls the body
                    # 'pose_hand': torch.Tensor(bdata['poses_60Hz'][:, 66:]).to(comp_device), # controls the finger articulation
                    'trans': torch.Tensor(bdata['trans_60Hz']).to(comp_device), # controls the global body position
                    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
                    #'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
                }
                # assert len(bdata['betas']) == num_betas
                body_pose_hand = bm[gender](**body_parms)
                
                # {k:v for k,v in body_parms.items() if k in [
                #     'pose_body', 'betas', 'pose_hand', 'root_orient', 'trans'
                #     ]})
                
                positions = c2c(body_pose_hand.Jtr)[:, :24].copy()
                assert positions.shape == (time_length, 24, 3)
                body_trans = bdata['trans']


                positions = np.stack([positions[..., 0], positions[..., 2], positions[..., 1]], axis=-1) # x, z, y according to AMASS
                # In 3dpw left and right limb sequence is inverted compared to AMASS! 
                positions = np.stack([positions[..., i, :] for i in [0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18, 21,20, 22,23]], axis=-2)

                positions = positions # already in  meters 

                output[split][seq_name] = positions.astype('float32')
                count[s] += 1
                nframes[s] += positions.shape[0]
                
    print(f"Processed total {sum(count)} sequences: {splits[0]}={count[0]}, {splits[1]}={count[1]}, {splits[2]}={count[2]}")
    print(f"Total number of frames: {sum(nframes)}: {splits[0]}={nframes[0]}, {splits[1]}={nframes[1]}, {splits[2]}={nframes[2]}")
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)        
    print('Done.')

    return output
    
    
    
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='3DPW Process Raw Data')

    parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')

    args = parser.parse_args()

    np.random.seed(0)
    
    actions = "all" 
    
    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
    
    batch_size = 128
    original_folder = "../datasets/raw/3DPW"
    annotations_folder = "../datasets/annotations/3DPW"
    smpl_body_models_path = "../datasets/annotations/AMASS/bodymodels"
    dataset_name = '3dpw'
    stride = 1
    augmentation = 0
    multimodal_threshold = 0.4
    FPS = 60
    
    # precompute data and statistics
    for task in ["hmp"]:
        if_consider_hip = False if task == "hmp" else True
        precomputed_folder_task = f"../datasets/processed/3DPW/{task}/"
        annotations_folder_task = os.path.join(annotations_folder, task)
        os.makedirs(precomputed_folder_task, exist_ok=True)
        
        if task == "hmp":
            obs_length = 30
            pred_length = 120
        else:
            obs_length = 30
            pred_length = 120
        
        output_npz_path = os.path.join(precomputed_folder_task, f"{OUTPUT_3D}.npz")

        if os.path.exists(output_npz_path):
            print('The npz dataset already (partially) exists at', output_npz_path)        
        else: 
            sequences = preprocess_dataset(original_folder, smpl_body_models_path=smpl_body_models_path, output_path=output_npz_path)    

        D3PWSkeletonVanilla = get_skeleton_class(motion_repr_type="SkeletonVanilla", dataset_name=dataset_name.lower())
        D3PWSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name=dataset_name.lower() )
        segments_path = os.path.join(annotations_folder_task, f"segments_test_zero_shot.csv")

        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mean_motion_test.txt")):
            print('Values for CMD already (partially) exists at', os.path.join(precomputed_folder_task, f"mean_motion_test.txt"))        
        else: 
            print("Computing values for CMD.")
            # below to compute the average motion per class for the TEST set, for the CMD computation
            def return_skeletons_for_stats_funct(current_task):
                def partialize_skel_class(skel_class):
                    return partial(skel_class, obs_length=obs_length, pred_length=pred_length)
                if current_task == "hmp":
                    skeletons = [partialize_skel_class(D3PWSkeletonCenterPose), None]
                else: 
                    skeletons = [partialize_skel_class(D3PWSkeletonVanilla), partialize_skel_class(D3PWSkeletonCenterPose)]
                return skeletons
            
            compute_stats(precomputed_folder=precomputed_folder_task, 
                          Dataset=partial(D3PWZeroShotDataset,annotations_folder=annotations_folder_task, actions=actions, stride=stride,  segments_path=segments_path, obs_length=obs_length, pred_length=pred_length),
                          Skeleton=return_skeletons_for_stats_funct(task),
                          if_consider_hip=if_consider_hip, 
                         batch_size=batch_size)
        if task == "hmp":
            skeleton= D3PWSkeletonCenterPose(if_consider_hip=if_consider_hip, obs_length=obs_length, pred_length=pred_length) #H36MSkeletonCenterPose
        else:
            assert 0, "Not implemented"
        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mmgt_test.txt")):
            print('Multimodal Ground Truth already (partially) exists at', os.path.join(precomputed_folder_task, f"mmgt_test.txt"))        
        else: 
            print(f"Computing Multimodal Ground Truth for task {task}.")
            if task == "hmp": 
                compute_multimodal_gt( Dataset=partial(D3PWZeroShotDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=if_consider_hip,
                                                    stride=stride),
                                    multimodal_threshold=multimodal_threshold, segment_path= segments_path, split='test',
                                    annotations_folder=annotations_folder_task, precomputed_folder=precomputed_folder_task,
                                        )
        
    
    
