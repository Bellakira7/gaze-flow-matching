from pathlib import Path
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

GAZELLE_ROOT = str(Path(__file__).resolve().parents[1] / 'third_party/gazelle')
DINOV2_ROOT = os.environ.get('DINOV2_ROOT', str(Path(__file__).resolve().parents[1] / 'third_party/dinov2'))
DINOV2_WEIGHTS = {
    'dinov2_vitl14': os.environ.get('DINOV2_WEIGHTS'),
}
FEAT_DIMS = {'dinov2_vitb14': 768, 'dinov2_vitl14': 1024}
IN_SIZE = (448, 448)

sys.path.insert(0, GAZELLE_ROOT)
import gazelle.utils as gutils  # noqa: E402
from gazelle.backbone import Backbone, DinoV2Backbone  # noqa: E402
from gazelle.model import GazeLLE  # noqa: E402


class LocalDinoV2Backbone(DinoV2Backbone):
    """DINOv2 from the vendored source; set DINOV2_WEIGHTS for offline weights. Transform and forward follow Gaze-LLE."""

    def __init__(self, model_name, weights=None):
        Backbone.__init__(self)
        weights = weights or DINOV2_WEIGHTS.get(model_name)
        self.model = torch.hub.load(DINOV2_ROOT, model_name, source='local', pretrained=not bool(weights))
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)


class CachedFeatureBackbone(Backbone):
    """恒等 backbone：input["images"] 直接就是 [B, C, h, w] 的冻结 DINOv2 特征，其余走官方 GazeLLE 不改。"""

    def __init__(self, dim, patch_size=14):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

    def forward(self, x):
        return x

    def get_dimension(self):
        return self.dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)


def build_cached_gazelle(backbone_name, inout=False):
    return GazeLLE(CachedFeatureBackbone(FEAT_DIMS[backbone_name]), inout=inout)


def gazefollow_l2_fixed(heatmap, gt_gazex, gt_gazey):
    """与官方 gazefollow_l2 完全一致，只把 argmax 格子换成格子中心 (idx+0.5)/size。"""
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (heatmap.shape[0], heatmap.shape[1]))
    pred_x = (pred_x + 0.5) / float(heatmap.shape[1])
    pred_y = (pred_y + 0.5) / float(heatmap.shape[0])
    gazex = np.array(gt_gazex)
    gazey = np.array(gt_gazey)
    avg_l2 = np.sqrt((pred_x - gazex.mean()) ** 2 + (pred_y - gazey.mean()) ** 2)
    min_l2 = np.sqrt((pred_x - gazex) ** 2 + (pred_y - gazey) ** 2).min().item()
    return avg_l2, min_l2


class _ImgSize:
    def __init__(self, width, height):
        self.size = (width, height)


class CachedGazeFollow(torch.utils.data.Dataset):
    """train: 每个 (图, 头) 一个样本，增强 = 水平翻转(换用翻转视图特征) + bbox 抖动，**没有随机裁剪**（裁剪会改变特征）。
    test: 每张图一个样本，含该图全部头，评测口径同官方 eval_gazefollow.py。"""

    def __init__(self, data_path, feat_dir, split, flip_view=True):
        self.data_path = data_path
        self.split = split
        self.data = json.load(open(os.path.join(data_path, f'{split}_preprocessed.json')))
        self.feat_dir = feat_dir
        self.paths = json.load(open(os.path.join(feat_dir, f'{split}_index.json')))
        assert self.paths == [d['path'] for d in self.data], 'feature index 与 preprocessed.json 不一致'
        self.has_flip = flip_view and os.path.exists(os.path.join(feat_dir, f'{split}_flip.npy'))
        self._mm = {}
        if split == 'train':
            self.idxs = [(i, j) for i, d in enumerate(self.data) for j, h in enumerate(d['heads']) if h['inout'] == 1]

    def _feat(self, view, row):
        if view not in self._mm:
            self._mm[view] = np.load(os.path.join(self.feat_dir, f'{self.split}_{view}.npy'), mmap_mode='r')
        return torch.from_numpy(np.array(self._mm[view][row]))

    def __len__(self):
        return len(self.idxs) if self.split == 'train' else len(self.data)

    def __getitem__(self, idx):
        if self.split == 'test':
            item = self.data[idx]
            bboxes = [h['bbox_norm'] for h in item['heads']]
            gazex = [h['gazex_norm'] for h in item['heads']]
            gazey = [h['gazey_norm'] for h in item['heads']]
            return self._feat('orig', idx), bboxes, gazex, gazey, item['height'], item['width']

        i, j = self.idxs[idx]
        item = self.data[i]
        head = item['heads'][j]
        width, height = item['width'], item['height']
        bbox = list(head['bbox'])
        gazex = list(head['gazex'])
        gazey = list(head['gazey'])
        view = 'orig'
        if self.has_flip and np.random.sample() <= 0.5:
            view = 'flip'
            xmin, ymin, xmax, ymax = bbox
            bbox = [width - xmax, ymin, width - xmin, ymax]
            gazex = [width - x for x in gazex]
        if np.random.sample() <= 0.5:
            bbox = gutils.random_bbox_jitter(_ImgSize(width, height), bbox)
        bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        gazex_norm = [x / float(width) for x in gazex]
        gazey_norm = [y / float(height) for y in gazey]
        heatmap = gutils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
        return self._feat(view, i), bbox_norm, heatmap


def collate_train(batch):
    feats, bboxes, heatmaps = zip(*batch)
    return torch.stack(feats), list(bboxes), torch.stack(heatmaps)


def collate_test(batch):
    feats, bboxes, gazex, gazey, height, width = zip(*batch)
    return torch.stack(feats), list(bboxes), list(gazex), list(gazey), list(height), list(width)


def _instance_metrics(args):
    heatmap, gx, gy, h, w = args
    torch.set_num_threads(1)
    hm = torch.from_numpy(heatmap)
    auc = gutils.gazefollow_auc(hm, gx, gy, h, w)
    avg_l2, min_l2 = gutils.gazefollow_l2(hm, gx, gy)
    avg_l2_fix, min_l2_fix = gazefollow_l2_fixed(hm, gx, gy)
    return auc, avg_l2, min_l2, avg_l2_fix, min_l2_fix


@torch.no_grad()
def evaluate(model, test_ds, device, batch_size=64, n_proc=32, n_workers=8, dump=None):
    """返回 dict：auc / avg_l2 / min_l2 为官方口径（argmax 无 +0.5），*_fix 为修正口径。"""
    model.eval()
    dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_test, num_workers=n_workers)
    jobs, meta = [], []
    row = 0
    for feats, bboxes, gazex, gazey, height, width in dl:
        preds = model({'images': feats.to(device).float(), 'bboxes': bboxes})
        for i in range(feats.shape[0]):
            for j in range(len(bboxes[i])):
                jobs.append((preds['heatmap'][i][j].float().cpu().numpy(), gazex[i][j], gazey[i][j], height[i], width[i]))
                meta.append((row + i, j))
        row += feats.shape[0]
    with Pool(n_proc) as pool:
        res = np.array(pool.map(_instance_metrics, jobs, chunksize=16))
    if dump:
        np.savez(dump, heatmaps=np.stack([j[0] for j in jobs]).astype(np.float16), meta=np.array(meta), per_instance=res)
    names = ['auc', 'avg_l2', 'min_l2', 'avg_l2_fix', 'min_l2_fix']
    out = {k: float(res[:, c].mean()) for c, k in enumerate(names)}
    out['n_instances'] = len(jobs)
    return out
