import cv2
import glob

concat_results = glob.glob('experiments/UNet_vessel_seg/result_img/Result_*.png')

for raw_result_path in concat_results:
    raw_result = cv2.imread(raw_result_path)
    h, w, _ = raw_result.shape
    h_, w_ = h, w // 4
    assert w_ * 4 == w

    orig = raw_result[:, 0*w_:1*w_, :]
    prob = raw_result[:, 1*w_:2*w_, :]
    mask = raw_result[:, 2*w_:3*w_, :]
    gt = raw_result[:, 3*w_:4*w_, :]

    cv2.imwrite(raw_result_path.replace('.png', '_orig.png'), orig)
    cv2.imwrite(raw_result_path.replace('.png', '_prob.png'), prob)
    cv2.imwrite(raw_result_path.replace('.png', '_mask.png'), mask)
    cv2.imwrite(raw_result_path.replace('.png', '_gt.png'), gt)
