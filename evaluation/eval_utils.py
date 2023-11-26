import torch
from torch import nn


class VideosData:
    id2vid = {
        11: "highway",
        12: "pedestrians",
        13: "office",
        14: "PETS2006",
        21: "badminton",
        22: "traffic",
        23: "boulevard",
        24: "sidewalk",
        31: "skating",
        32: "blizzard",
        33: "snowFall",
        34: "wetSnow",
        41: "boats",
        42: "canoe",
        43: "fall",
        44: "fountain01",
        45: "fountain02",
        46: "overpass",
        51: "abandonedBox",
        52: "parking",
        53: "sofa",
        54: "streetLight",
        55: "tramstop",
        56: "winterDriveway",
        61: "port_0_17fps",
        62: "tramCrossroad_1fps",
        63: "tunnelExit_0_35fps",
        64: "turnpike_0_5fps",
        71: "bridgeEntry",
        72: "busyBoulvard",
        73: "fluidHighway",
        74: "streetCornerAtNight",
        75: "tramStation",
        76: "winterStreet",
        81: "continuousPan",
        82: "intermittentPan",
        83: "twoPositionPTZCam",
        84: "zoomInZoomOut",
        91: "backdoor",
        92: "bungalows",
        93: "busStation",
        94: "copyMachine",
        95: "cubicle",
        96: "peopleInShade",
        101: "corridor",
        102: "diningRoom",
        103: "lakeSide",
        104: "library",
        105: "park",
        111: "turbulence0",
        112: "turbulence1",
        113: "turbulence2",
        114: "turbulence3",
    }
    vid2id = {v: k for k, v in id2vid.items()}

    id2cat = {
        1: "baseline",
        2: "cameraJitter",
        3: "badWeather",
        4: "dynamicBackground",
        5: "intermittentObjectMotion",
        6: "lowFramerate",
        7: "nightVideos",
        8: "PTZ",
        9: "shadow",
        10: "thermal",
        11: "turbulence",
    }
    cat2id = {v: k for k, v in id2cat.items()}

    vid_matrix: dict[int : torch.Tensor] = dict()

    @classmethod
    def save_result(cls, result: torch.Tensor) -> None:
        for feature in result:
            vid_indx = feature[-1].item()
            if cls.vid_matrix.get(vid_indx, None) is None:
                cls.vid_matrix[vid_indx] = torch.zeros(4, dtype=torch.int32, device="cpu")
            cls.vid_matrix[vid_indx] += feature[:-1]

    @classmethod
    def get_vid_ratio(cls, vid: str | int):
        ''' output: torch.tensor(batch, channel, 1, (precision, recall ,fnr, f_score, accuracy))'''
        if type(vid) == str:
            vid = cls.vid2id.get(vid)
            if vid is None:
                print(f"Error | vid: {vid}")
                exit()
        tp, fp, tn, fn = cls.vid_matrix[vid]
        prec = tp / (tp + fp) if (tp + fp) != 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) != 0 else float("nan")
        fnr = 1 - recall

        f_score = (
            torch.tensor(1)
            if (tp + fn) == 0
            else torch.tensor(0)
            if tp == 0
            else 2 * (prec * recall) / (prec + recall)
        )
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn)!= 0 else float('nan')

        return torch.tensor((prec, recall, fnr, f_score, acc), dtype=torch.float32)

    @classmethod
    def get_cat_ratio(cls, cat: str | int):
        if type(cat) == str:
            cat = cls.cat2id.get(cat)
            if cat is None:
                print(f"Error | cat: {cat}")
                exit()
        
        results = torch.zeros(5, dtype=torch.float32)
        
        for i, vid in enumerate(cls.id2vid.keys(), 1 ):
            if vid // 10 == cat:
                ratios = cls.get_vid_ratio(vid) 
                result = (result*(i-1) + ratios)/i
        return results
    
class EvalMeasure(nn.Module):
    def __init__(self, thresh: float):
        super().__init__()
        self.thresh = thresh

    def forward(
        self, gts: torch.Tensor, preds: torch.Tensor, vid_indices: torch.Tensor
    ) -> torch.Tensor:
        '''
        gts, preds : 4-dimension -> batch * channel * im_height * im_width. (batch, channel, height, width)
        vid_indices: 2-dimension -> batch * 1. (batch, vid_indx)
        result : 2-dimension -> batch * 5;  (batch, features) ; features-> (tp, fp, tn, fn, vid_indx)
        '''
        gts = gts.to(dtype= torch.int32)
        preds = torch.where(preds > self.thresh, 1, 0).to(dtype=torch.int32)
        
        diff = gts ^ preds
        same = diff ^ torch.ones_like(diff)
        gt_1 = gts == 1
        gt_0 = gts == 0
        
        tp, fp = gt_1 * same ,  gt_0 *diff
        tn, fn = gt_0 * same , gt_1 * diff
        batch = tp.size()[0]
        
        tp, fp = tp.view(batch, -1).sum(dim=1,keepdim=True), fp.view(batch, -1).sum(dim=1, keepdim=True)
        tn, fn = tn.view(batch, -1).sum(dim=1, keepdim=True), fn.view(batch, -1).sum(dim=1, keepdim=True)
        
        result = torch.cat((tp, fp, tn, fn, vid_indices), dim=1).cpu()
        return result



if __name__ == "__main__":
    import os
    import sys
    import cv2 as cv 
    import numpy as np
    from pathlib import Path
    import torchvision.transforms as tvtf
    
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.data_preprocess import CDNet2014Preprocess
    
    preprocess = CDNet2014Preprocess((224,224))
    eval = EvalMeasure(0.5)
    gts_pth = "/root/Work/fork-BGS/BSUV-Net-2.0/dataset/currentFr/baseline/highway/groundtruth"
    imgs_name = sorted(os.listdir(gts_pth))
    
    gts = list()
    vid_indices= list()
    for i, img_name in enumerate(imgs_name[699:]):
        if img_name.split('.')[-1] != 'png':
            continue
        elif i == 5:
            break
        
        img = cv.imread(os.path.join(gts_pth, img_name), cv.IMREAD_GRAYSCALE)
        gt = preprocess(img)
        gts.append(tvtf.ToTensor()(gt.copy()))
        vid_indices.append(11)
        # gt[gt == 1] = 255
        # gt[gt == -1] = 128
        # gt[gt == 0] = 0
        # cv.imwrite(f'./{i+1+699}.png', gt)
    

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    gts = torch.cat(gts, dim=0).reshape(len(gts),1,224,224).to(device= device)
    preds = torch.zeros_like(gts, device= device)
    vid_indices = torch.tensor(vid_indices).reshape(5,1).to(device= device, dtype=torch.int32)

    with torch.no_grad():
        result = eval(gts, preds, vid_indices)
    print(result)
    VideosData.save_result(result)
    print(VideosData.get_vid_ratio(11))