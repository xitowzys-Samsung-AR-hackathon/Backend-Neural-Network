import torch
import torch.nn.functional as F

import warnings
import os

warnings.filterwarnings('ignore')

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import glob
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import numpy as np
import cv2
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
from PIL import Image

os.environ['MEGADEPTH'] = 'MEGADEPTH_LOCATION'

from options.options import get_dataset, get_model
import models.architectures as architectures

import matplotlib.pyplot as plt
import os
# from google.colab import files

from utils.visualization_utils import *

import argparse
from pyinstrument import Profiler
from line_profiler import LineProfiler

import cProfile, pstats, io
from pstats import SortKey


# from google.colab import files


class DataBase:
    def __init__(self, model):
        self.model = model
        self.targets = []
        self.processed_targets = []
        self.filepaths = []

        self.device = 'cpu'
        if not torch.cuda.is_available():
            pass
        else:
            self.device = 'cuda'

        self.num_of_rot = 2

    def add_target(self, path):
        assert isfile(path)

        target = Image.open(path).convert("RGB")
        self.filepaths.append(path)

        for i in range(self.num_of_rot):
            self.targets.append(target.rotate(i * 90, expand=True))
            self.processed_targets.append(self.model.transform(self.targets[-1]))

    def add_targets(self, path):
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        for file_name in files:
            self.add_target(file_name)

    # def upload_targets(self, folder_path='targets'):
    #   database_name = folder_path # replace this with the actual path to your folder
    #   os.makedirs(f"./{database_name}", exist_ok=True)
    #   # Use files.upload() to upload all the files in the foldern
    #   uploaded = files.upload()
    #
    #   for name, data in uploaded.items():
    #     file_path = f"./{database_name}/{name}"
    #     with open(file_path, 'wb') as f:
    #           f.write(data)
    #
    #     self.add_target(file_path)

    def get_best_matching(self, photo_path, batch_size=4):
        assert isfile(photo_path)

        targets = self.processed_targets

        photo = Image.open(photo_path).convert("RGB")
        photo = self.model.transform(photo)

        scores = []

        ids1 = torch.arange(0, 16384, device=self.device).expand(batch_size, 16384)

        for i in range(0, len(targets), batch_size):
            batchTarget = targets[i:i + batch_size]
            batchPhoto = [photo] * len(batchTarget)

            batchTarget = torch.stack(batchTarget)
            batchPhoto = torch.stack(batchPhoto)

            scores.extend(self.model.get_pair_scores(batchPhoto, batchTarget, ids1))

        best_matching_id = np.argmax(scores)
        best_score = scores[best_matching_id]

        best_matching_id = best_matching_id // self.num_of_rot * self.num_of_rot

        return best_matching_id, best_score


class CoamModel:
    def __init__(self, path):
        assert isfile(path)

        self.device = 'cpu'
        if torch.cuda.is_available() == False:
            old_model = torch.load(path, map_location=torch.device('cpu'))
        else:
            old_model = torch.load(path)
            self.device = 'cuda'

        opts = old_model['opts']
        opts.W = 128

        model = get_model(opts)
        model.load_state_dict(old_model['state_dict'])
        model.to(self.device)

        model = model.eval()

        transform = transforms.Compose([
            transforms.Resize((opts.W, opts.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.model = model
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def set_model(self, model):
        self.model = model

    def best_matches(self, sim, cond1=None, cond2=None, topk=8000, T=0.3, nn=1, is_batch=False):
        ''' Find the best matches for a given NxN matrix.
            Optionally, pass in the actual indices corresponding to this matrix
            (cond1, cond2) and update the matches according to these indices.
        '''

        if is_batch:
            nn12 = torch.max(sim, dim=2)[1]
            nn21 = torch.max(sim, dim=1)[1]

            ids1 = torch.arange(0, sim.shape[1]).expand(sim.shape[0], sim.shape[1]).to(nn12.device)
            mask = (ids1 == nn21[np.arange(nn12.shape[0])[:, None], nn12])
            best_matches_batch = []

            for i in range(mask.shape[0]):
                best_matches = torch.stack([ids1[i, mask[i]], nn12[i, mask[i]]])
                preds = sim[i, ids1[i, mask[i]], nn12[i, mask[i]]]
                res, ids = preds.sort()
                ids = ids[res > T]
                best_matches = best_matches[:, ids[-topk:]]
                best_matches_batch.append(best_matches.t())

            return best_matches_batch

        else:
            nn12 = torch.max(sim, dim=1)[1]
            nn21 = torch.max(sim, dim=0)[1]

            ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
            mask = (ids1 == nn21[nn12])

            matches = torch.stack([ids1[mask], nn12[mask]])

            preds = sim[ids1[mask], nn12[mask]]
            res, ids = preds.sort()
            ids = ids[res > T]

            if not (cond1 is None) and not (cond2 is None):
                cond_ids1 = cond1[ids1[mask]]
                cond_ids2 = cond2[nn12[mask]]

                matches = torch.stack([cond_ids1, cond_ids2])

            matches = matches[:, ids[-topk:]]

            return matches.t()

    def get_matches(self, im1_orig, im2_orig):
        im1 = self.transform(im1_orig)
        im2 = self.transform(im2_orig)

        W1, H1 = im1_orig.size
        W2, H2 = im2_orig.size

        im1 = im1.to(self.device)
        im2 = im2.to(self.device)

        # Now visualise heat map for a given point
        with torch.no_grad():
            results = self.model.run_feature((im1.unsqueeze(0), None), (im2.unsqueeze(0), None),
                                             MAX=16384, keypoints=None, sz1=(128, 128), sz2=(128, 128),
                                             factor=1, r_im1=(1, 1), r_im2=(1, 1),
                                             use_conf=1, T=0.1,
                                             return_4dtensor=True, RETURN_ALLKEYPOINTS=True)

        matches = self.best_matches(results['match'].view(128 * 128, 128 * 128), topk=100, T=0.6)
        _, idxs = matches[:, 0].sort(dim=0)
        matches = matches[idxs, :]
        matches = matches[1::(100 // 20), :][:, :]

        score = len(matches) / 20

        WG = 128;
        RG = 1

        kps1 = results['kp1'].clone()
        kps1[:, 0] = (results['kp1'][:, 0] / (1 * 2.) + 0.5) * W1
        kps1[:, 1] = (results['kp1'][:, 1] / (1 * 2.) + 0.5) * H1

        kps2 = results['kp2'].clone()
        kps2[:, 0] = (results['kp2'][:, 0] / (1 * 2.) + 0.5) * W2
        kps2[:, 1] = (results['kp2'][:, 1] / (1 * 2.) + 0.5) * H2

        return kps1[matches[:, 0], :], kps2[matches[:, 1], :], score

    def get_similarity_scores(self, sim, ids1):
        n12 = torch.max(sim, dim=1)[0]
        n12, ids = n12.sort()
        return n12[:, -100:].mean(dim=1)

    # def get_similarity_scores(self, sim, ids1):
    #   nn12 = torch.max(sim, dim=2)[1]
    #   nn21 = torch.max(sim, dim=1)[1]

    #   mask = torch.arange(nn12.shape[0])[:,None]
    #   mask = nn21[mask,nn12]
    #   mask = (ids1 == mask)
    #   sim_scores = []

    #   for i in range(mask.shape[0]):
    #     preds = sim[i, ids1[i,mask[i]], nn12[i,mask[i]]]
    #     res, ids = preds.sort()
    #     sim_scores.append(res[-100:].mean())        

    #   return sim_scores

    # def get_similarity_scores(self, sim, ids1):
    #   nn12 = np.asarray(torch.max(sim, dim=2)[1].cpu())
    #   nn21 = np.asarray(torch.max(sim, dim=1)[1].cpu())
    #   ids1 = np.asarray(ids1.cpu())

    #   mask = np.equal(ids1, nn21[torch.arange(nn12.shape[0])[:,None],nn12])
    #   sim_scores = []

    #   for i in range(mask.shape[0]):
    #     preds = sim[i, ids1[i,mask[i]], nn12[i,mask[i]]]
    #     res, ids = preds.sort()
    #     sim_scores.append(res[-100:].mean())        

    #   return sim_scores

    def get_pair_scores(self, im1_orig, im2_orig, ids1):
        im1 = im1_orig.to(self.device)
        im2 = im2_orig.to(self.device)

        with torch.no_grad():
            results = self.model.run_feature((im1, None), (im2, None),
                                             MAX=16384, use_conf=True, sz1=(128, 128), sz2=(128, 128))

        if len(results['match'].shape) == 2:
            results['match'] = results['match'].view(-1, 128 * 128, 128 * 128)

        scores = self.get_similarity_scores(results['match'], ids1)
        scores = [x.item() for x in scores]

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",
                        "-m",
                        type=str,
                        default='./pretrainedmodels/effnetB1_ep86.pth')
    parser.add_argument("--targets-path",
                        "-t",
                        type=str,
                        default='./demo/targets')
    parser.add_argument("--photo-path", "-p", type=str, default="./demo/imgs/photo_1.png")
    parser.add_argument("--profiler", "-prof", default="None", type=str)

    args = parser.parse_args()

    # Initilize model
    coamModel = CoamModel(args.model_path)

    # Initilize database and populate it with targets
    dataBase = DataBase(coamModel)

    if args.profiler == "pyinstrument":
        profiler = Profiler()
        profiler.start()
    elif args.profiler == "cProfiler":
        pr = cProfile.Profile()
        pr.enable()
    elif args.profiler == "line":
        lprofiler = LineProfiler()
        lprofiler.add_function(coamModel.get_similarity_scores)
        lprofiler.add_function(coamModel.get_pair_scores)
        lp_wrapper = lprofiler(dataBase.get_best_matching)

    dataBase.add_targets(args.targets_path)

    # Load photo and get target that best matches it
    if args.profiler != "line":
        target_id, score = dataBase.get_best_matching(args.photo_path)
    else:
        target_id, score = lp_wrapper(args.photo_path)
        lprofiler.print_stats(output_unit=0.001)

    if args.profiler == "pyinstrument":
        profiler.stop()
        output = profiler.output_html()
        fp = open("profilerOutput.html", "w")
        fp.write(output)
        fp.close()
    elif args.profiler == "cProfiler":
        pr.disable()
        pr.dump_stats("coam_stats")

    print(dataBase.filepaths[target_id // dataBase.num_of_rot], score)
