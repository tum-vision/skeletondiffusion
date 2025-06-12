import torch
import numpy as np

from .base import Kinematic


class AMASSKinematic(Kinematic):
    def __init__(self, num_joints=22, **kwargs):
        
        super().__init__(**kwargs)
        self.relative_restpose  = {'left_arm': np.array([[-0.08951462,  0.01432468,  0.11159658],
       [-0.116652  , -0.01674555,  0.01757753],
       [-0.02959993,  0.0193723 , -0.2667216 ],
       [ 0.03015181, -0.12220491, -0.25177056]]), 
                        'right_arm': np.array([[ 0.05214295,  0.01628557,  0.11822093],
       [ 0.11756767,  0.01566032,  0.03865051],
       [ 0.04485199,  0.02369931, -0.26253664],
       [ 0.00150388, -0.10840166, -0.25212884]]),
                        'left_leg': np.array([[-0.00102365,  0.03786719, -0.37976843],
       [-0.05257663,  0.00471593, -0.3990224 ],
       [ 0.04243585, -0.1181156 , -0.04386783]]), 
                        'right_leg': np.array([[-0.00650667,  0.01321159,  0.10511905],
       [ 0.01109964, -0.02364293,  0.13600564],
       [ 0.003884  , -0.04299295,  0.03277576]])}
        
        self.joint_dict_orig  = {0: 'GlobalRoot', 1: 'RHip', 2: 'LHip', 3: 'Spine1',       
                                 4: 'RKnee', 5: 'LKnee', 6: 'Spine3', 
                                 7: 'RHeel', 8: 'LHeel', 9: 'Neck', 10: 'RFoot', 
                                 11: 'LFoot', 12: 'BMN', 13: 'RSI',        14: 'LSI', 15: 'Head',
                                 16: 'RShoulder', 17: 'LShoulder', 18: 'RElbow',19: 'LElbow', 20: 'RWrist',21: 'LWrist'}
        
        self.limbseq = [ [0,3], [3,6], [6,9], [9,12], [12,15], # body
                        [9,14], [14,17], [17,19], [19,21], # right arm
                        [9,13], [13,16], [16,18], [18,20], # left arm
                        [0,2], [2,5], [5,8], [8,11], # right leg
                        [0,1], [1,4], [4,7], [7,10] # left leg
                    ]
        
        assert num_joints in [22, 52]
        if num_joints == 52:
            self.joint_dict_orig = {**self.joint_dict_orig, **{22: 'left_index1', 23: 'left_index2',  24: 'left_index3',
                                                                25: 'left_middle1', 26: 'left_middle2', 27: 'left_middle3',
                                                                28: 'left_pinky1', 29: 'left_pinky2', 30: 'left_pinky3',
                                                                31: 'left_ring1',  32: 'left_ring2', 33: 'left_ring3',
                                                                34: 'left_thumb1', 35: 'left_thumb2', 36: 'left_thumb3',
                                                                37: 'right_index1', 38: 'right_index2', 39: 'right_index3',
                                                                40: 'right_middle1', 41: 'right_middle2', 42: 'right_middle3',
                                                                43: 'right_pinky1', 44: 'right_pinky2', 45: 'right_pinky3',
                                                                46: 'right_ring1', 47: 'right_ring2', 48: 'right_ring3',
                                                                49: 'right_thumb1', 50: 'right_thumb2', 51: 'right_thumb3'
                                                                }}
            self.limbseq +=  [ [20, 22], [20, 25], [20, 28], [20, 31], [20,34],
                                [22,23], [23,24], [25,26], [26,27], [28,29], [29,30], [31,32], [32,33], [34,35], [35,36], # falanges left hand
                                [21,37], [21,40],[21,43], [21,46], [21,49],
                                [37,38], [38,39],  [40,41], [41,42],  [43,44], [44,45],  [46,47], [47,48],  [49,50], [50,51], # falanges right hand
                            ]     


        self.left_right_limb_list = [False if (joint[0] == "L" and joint[1].isupper()) or 'left' in joint else True for joint in list(self.joint_dict_orig.values())]
            
        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i:v for i,v in enumerate(list(self.node_dict.values()))}
            node_dict_reversed = {v:i for i,v in self.node_dict.items()}
            self.node_limbseq = [[node_dict_reversed['RHip'], node_dict_reversed['LHip']], 
                                    [node_dict_reversed['RHip'], node_dict_reversed['Spine1']], 
                                    [node_dict_reversed['LHip'], node_dict_reversed['Spine1']],
                                    *[[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]]
            # # index of limbs in limb_angles_idx for mae
            # index of joints in kinchain for limb dropping
            self.limb_angles_idx = [[0,2,3,4,5,6], [0,3], [4,7,8,9,10], [4,11,12,13,14], [0,15,16,17], [18,19,20]] 
            self.kinchain = [[2,5,8,11,14], #hip to head
                            [8,13,16,18,20], # left arm
                            [8,12,15,17,19], # right arm
                            [1,4,7,10], # left leg
                            [0,3,6,9], # right leg
                            [0,1,2,0], # hip connection
                            ]
        else: 
            self.node_dict = {k:v for k,v in enumerate(list(self.node_hip.values())+list(self.joint_dict_orig.values())[1:])}
            self.node_limbseq = self.limbseq.copy()
            # self.limb_angles_idx = [[0,1,2,3,4], [5,6,7,8], [9,10,11,12], [13, 14, 15, 16], [17, 18, 19, 20]]
        self.limbseq = np.array(self.limbseq)

