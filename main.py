import numpy as np
from utils import generate_straightened_3d_scpr, curve_smooth_and_resample




if __name__ == "__main__":
    H, W, D = 512, 512, 640
    L = 500
    centerline_pts = np.random.randn((L, 3))
    ccta_img = np.random.randn((H, W, D))
    spacing = np.array([0.25, 0.25])
    pt_after_smooth, d1_, axis1, axis2 = curve_smooth_and_resample(centerline_pts / spacing, spacing, 0, 0,)
    scpr, grid_ori = generate_straightened_3d_scpr(
            ccta_img, spacing, pt_after_smooth * spacing, axis1, axis2, theta=0,
            sample_width=48,
            sample_unit=0.4,)