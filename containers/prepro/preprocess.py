import os
import shutil
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

input_dir = '/opt/ml/processing/input'
output_dir = '/opt/ml/processing/output'


class PennFudanDataset():
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        
def change_image(opr_name, ds, output_dir, factor=None):
    if opr_name == "cp":
        operator = None
    elif opr_name == "color":
        operator = ImageEnhance.Color
    elif opr_name == "contrast":
        operator = ImageEnhance.Contrast
    elif opr_name == "brightness":
        operator = ImageEnhance.Brightness
    elif opr_name == "sharpness":
        operator = ImageEnhance.Sharpness
    else:
        raise ValueError
                
    for img_name, mask_name in zip(ds.imgs, ds.masks):
        #! パスの整理
        ##! 既存パス
        img_path = Path(ds.root, "PNGImages", img_name)
        img_stem = img_path.stem
        img_suffix = img_path.suffix
        
        mask_path = Path(ds.root, "PedMasks", mask_name)
        mask_stem = mask_path.stem
        mask_suffix = mask_path.suffix
        
        ##! 新規パス
        aug_img_stem = img_stem + "_" + opr_name + "_" + str(factor)
        aug_img_name = aug_img_stem + img_suffix
        aug_img_path = Path(output_dir, "PNGImages", aug_img_name)
        
        aug_mask_stem = mask_stem + "_" + opr_name + "_" + str(factor)
        aug_mask_name = aug_mask_stem + img_suffix
        aug_mask_path = Path(output_dir, "PedMasks", aug_mask_name)
        
        #! copy用分岐
        if opr_name == "cp":
            shutil.copy(img_path, aug_img_path)
            shutil.copy(mask_path, aug_mask_path)
        else:
            #! PNGImageの加工
            img = Image.open(img_path)
            # img.show()
            enhancer = operator(img)
            img_enhance = enhancer.enhance(factor)
            img_enhance.save(aug_img_path)

            #! PedMaskのコピー
            shutil.copy(mask_path, aug_mask_path)
        
    return
        
        
def main():
    dataset = PennFudanDataset(input_dir)
    
    # create output dirs
    Path(output_dir).mkdir(exist_ok=True)
    Path(output_dir, "PNGImages").mkdir(exist_ok=True)
    Path(output_dir, "PedMasks").mkdir(exist_ok=True)
    
    # augment images
    for factor in [0.5, 0.75, 1.25, 1.5, 2.0]:
        change_image("color", dataset, output_dir, factor)
        change_image("brightness", dataset, output_dir, factor)

    # copy original images
    change_image("cp", dataset, output_dir)
    
    print('Finished running processing job')


if __name__ == "__main__":
    main()