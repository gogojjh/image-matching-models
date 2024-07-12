import sys
from pathlib import Path
import os
import torchvision.transforms as tfm
import py3_wget

sys.path.append(str(Path(__file__).parent.parent.joinpath("third_party/duster")))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from matching.base_matcher import BaseMatcher
from matching.utils import to_numpy, resize_to_divisible

from matching import WEIGHTS_DIR


class Dust3rMatcher(BaseMatcher):
    model_path = WEIGHTS_DIR.joinpath("duster_vit_large.pth")
    vit_patch_size = 16

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        if not os.path.isfile(Dust3rMatcher.model_path):
            print("Downloading Dust3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, Dust3rMatcher.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.vit_patch_size)

        img = self.normalize(img).unsqueeze(0)

        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        images = [
            {"img": img0, "idx": 0, "instance": 0},
            {"img": img1, "idx": 1, "instance": 1},
        ]
        pairs = make_pairs(
            images, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(
            pairs, self.model, self.device, batch_size=1, verbose=self.verbose
        )

        '''
        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']
        '''

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PairViewer,
            verbose=self.verbose,
        )
        self.scene = scene

        # retrieve useful values from scene:
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        # poses = scene.get_im_poses()
        # focals = scene.get_focals()

        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(
                xy_grid(*imgs[i].shape[:2][::-1])[conf_i]
            )  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)

        mkpts1 = pts2d_list[1][reciprocal_in_P2]
        mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
