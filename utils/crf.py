import pydensecrf.densecrf as dcrf
import pydensecrf.utils
import numpy as np

class DenseCRF():
    def __init__(self, iter_max=10, bi_w=7, bi_xy_std=50, bi_rgb_std=4, pos_w=3, pos_xy_std=3):
        self.iter_max = iter_max
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std

    def __call__(self, image, prob_map):
        C, H, W = prob_map.shape
        
        image = image.permute((1, 2, 0)).detach().cpu().numpy()
        image = (image*255).astype(np.ubyte)
        prob_map = prob_map.detach().cpu().numpy()
        
        U = pydensecrf.utils.unary_from_softmax(prob_map)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w)

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q