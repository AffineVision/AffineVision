import cv2

from bestvision.transforms import matrix2d
from bestvision.transforms import warp
from bestvision import assets


if __name__ == "__main__":
    image = cv2.imread(assets.lena)
    h, w = image.shape[:2]
    cv2.imshow("origin", image)

    # scale
    scale = matrix2d.scale([0.5, 2])

    img_scaled = warp.affine(image, scale, (w, h))

    cv2.imshow("scaled", img_scaled)

    # center rotate

    crs = matrix2d.center_rotate_scale_cw((w/2, h/2), 30, 1.25)

    img_crs = warp.affine(image, crs, (w, h))

    cv2.imshow("crs", img_crs)

    # hflip

    hflip = matrix2d.hflip(w)
    img_hfliped = warp.affine(image, hflip, (w, h))
    cv2.imshow("hflip", img_hfliped)

    vflip = matrix2d.vflip(h)
    img_vfliped = warp.affine(image, vflip, (w, h))
    cv2.imshow("vflip", img_vfliped)

    # shear

    shear_x = matrix2d.shear(0.5, 0)
    img_xsheared = warp.affine(image, shear_x, (w, h))
    cv2.imshow("xshear", img_xsheared)

    shear_y = matrix2d.shear(0, 0.5)
    img_ysheared = warp.affine(image, shear_y, (w, h))
    cv2.imshow("yshear", img_ysheared)

    cv2.waitKey()