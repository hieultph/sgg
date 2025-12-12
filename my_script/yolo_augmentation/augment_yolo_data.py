"""
YOLO Data Augmentation Script
Tăng cường dữ liệu cho YOLO dataset bằng Albumentations

Usage:
    python augment_yolo_data.py --input custom_yolo_data --output custom_yolo_data_augmented --augment-factor 5
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import yaml


class YOLODataAugmenter:
    def __init__(self, input_dir, output_dir, augment_factor=5):
        """
        Args:
            input_dir: Thư mục chứa YOLO dataset gốc
            output_dir: Thư mục lưu augmented dataset
            augment_factor: Số lượng ảnh augmented cho mỗi ảnh gốc
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augment_factor = augment_factor
        
        # Define comprehensive augmentation pipeline with detailed descriptions
        self.transform = A.Compose([
            # ==================== GEOMETRIC TRANSFORMATIONS ====================
            # Lật ảnh theo chiều ngang (trái <-> phải)
            # Use case: Tăng robustness cho objects có thể xuất hiện ở bất kỳ hướng nào
            A.HorizontalFlip(p=0.5),
            
            # Lật ảnh theo chiều dọc (trên <-> dưới)
            # Use case: Hữu ích cho aerial/satellite images hoặc objects không có hướng cố định
            # A.VerticalFlip(p=0.2),
            
            # Transpose ảnh (hoán đổi width/height, giống rotate 90° + flip)
            # Use case: Tạo biến thể khác cho objects có thể quay 90°
            # A.Transpose(p=0.2),
            
            # Shift (dịch), Scale (phóng to/thu nhỏ), Rotate (xoay) kết hợp
            # Use case: Simulate camera movements và thay đổi khoảng cách
            # A.ShiftScaleRotate(
            #     shift_limit=0.1,      # Dịch tối đa 10% image size
            #     scale_limit=0.2,      # Phóng to/thu nhỏ 0.8x-1.2x
            #     rotate_limit=20,      # Xoay ±20 độ
            #     border_mode=cv2.BORDER_CONSTANT, 
            #     p=0.6
            # ),
            
            # Affine transformation: scale, translate, rotate, shear (nghiêng)
            # Use case: Simulate góc nhìn camera khác nhau, perspective changes
            # A.Affine(
            #     scale=(0.8, 1.2),           # Zoom in/out
            #     translate_percent=(-0.1, 0.1),  # Dịch chuyển
            #     rotate=(-15, 15),           # Xoay
            #     shear=(-10, 10),           # Biến dạng nghiêng (giống nhìn từ góc chéo)
            #     p=0.3
            # ),
            
            # Perspective transform (biến đổi phối cảnh)
            # Use case: Simulate nhìn object từ các góc độ khác nhau (3D -> 2D projection)
            # A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Crop ngẫu nhiên và resize về size gốc
            # Use case: Simulate zoom và làm model focus vào các phần khác nhau của image
            # A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.4),
            
            # Elastic deformation (biến dạng đàn hồi - giống kéo dãn vải)
            # Use case: Simulate deformations, hữu ích cho flexible objects
            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            
            # Grid distortion (biến dạng lưới - uốn cong hình như lưới)
            # Use case: Simulate lens distortions, fisheye effect
            # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            
            # Optical distortion (méo quang học - barrel/pincushion distortion)
            # Use case: Simulate camera lens distortions
            # A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.2),
            
            # ==================== COLOR TRANSFORMATIONS ====================
            # Thay đổi độ sáng và độ tương phản
            # Use case: Simulate different lighting conditions (sáng/tối, high/low contrast)
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            
            # Color space transformations (chọn 1 trong 3)
            A.OneOf([
                # HSV: Thay đổi Hue (màu sắc), Saturation (độ bão hòa), Value (độ sáng)
                # Use case: Simulate color variations, different white balance
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=1.0),
                
                # RGB Shift: Thay đổi từng kênh màu R, G, B
                # Use case: Simulate color cast, lighting color changes
                # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                
                # Color Jitter: Kết hợp nhiều thay đổi màu
                # Use case: General color augmentation
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.5),
            
            # Gamma correction (điều chỉnh gamma curve)
            # Use case: Simulate different exposure levels, screen brightness
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Tone curve adjustment (điều chỉnh đường cong tone)
            # Use case: Simulate film/camera tone curves, color grading
            A.RandomToneCurve(scale=0.3, p=0.3),
            
            # Histogram equalization (cân bằng histogram)
            # Use case: Enhance contrast, especially in low-contrast images
            A.Equalize(p=0.2),
            
            # CLAHE: Contrast Limited Adaptive Histogram Equalization
            # Use case: Improve local contrast, enhance details in shadows/highlights
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            
            # Posterize (giảm số lượng màu - hiệu ứng poster)
            # Use case: Simulate low bit-depth images, compressed images
            # A.Posterize(num_bits=4, p=0.2),
            
            # Solarize (đảo ngược màu ở vùng sáng)
            # Use case: Simulate overexposure effects
            # A.Solarize(threshold=128, p=0.2),
            
            # Channel Shuffle (hoán đổi vị trí channels RGB)
            # Use case: Force model không phụ thuộc vào thứ tự channels
            # A.ChannelShuffle(p=0.2),
            
            # Invert (đảo ngược màu - negative image)
            # Use case: Extreme augmentation, force model học features không phụ thuộc màu
            # A.InvertImg(p=0.1),
            
            # ==================== NOISE & BLUR ====================
            # Noise types (chọn 1 trong 3)
            # A.OneOf([
            #     # Gaussian noise (nhiễu Gaussian - random noise)
            #     # Use case: Simulate sensor noise, low-light conditions
            #     A.GaussNoise(var_limit=(20.0, 80.0), p=1.0),
                
            #     # ISO noise (nhiễu ISO - giống camera ISO cao)
            #     # Use case: Simulate high ISO photography, low-light with color noise
            #     A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                
            #     # Multiplicative noise (nhiễu nhân - speckle noise)
            #     # Use case: Simulate certain types of sensor noise
            #     A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
            # ], p=0.4),
            
            # Blur types (chọn 1 trong 5)
            # A.OneOf([
            #     # Gaussian blur (làm mờ Gaussian - natural blur)
            #     # Use case: Simulate out-of-focus, depth of field
            #     A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                
            #     # Motion blur (làm mờ chuyển động)
            #     # Use case: Simulate camera shake, moving objects
            #     A.MotionBlur(blur_limit=7, p=1.0),
                
            #     # Median blur (làm mờ trung vị - preserves edges better)
            #     # Use case: Reduce noise while keeping edges
            #     A.MedianBlur(blur_limit=7, p=1.0),
                
            #     # Average blur (làm mờ trung bình - uniform blur)
            #     # Use case: Simple blur effect
            #     A.Blur(blur_limit=5, p=1.0),
                
            #     # Advanced blur (blur nâng cao với bokeh effect)
            #     # Use case: Simulate camera bokeh, lens blur
            #     A.AdvancedBlur(blur_limit=(3, 7), p=1.0),
            # ], p=0.4),
            
            # Defocus blur (làm mờ mất focus)
            # Use case: Simulate out-of-focus areas, depth of field
            # A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=0.2),
            
            # Zoom blur (làm mờ zoom - radial blur)
            # Use case: Simulate zoom motion, speed effect
            # A.ZoomBlur(max_factor=1.05, p=0.2),
            
            # Glass blur (làm mờ kính - nhìn qua kính)
            # Use case: Simulate looking through textured glass
            # A.GlassBlur(sigma=0.7, max_delta=2, iterations=2, p=0.1),
            
            # ==================== WEATHER & LIGHTING ====================
            # Random fog (thêm sương mù)
            # Use case: Simulate foggy weather conditions
            # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
            
            # Random rain (thêm mưa)
            # Use case: Simulate rainy conditions, water droplets
            # A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.2),
            
            # Random snow (thêm tuyết)
            # Use case: Simulate snowy weather
            # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.1),
            
            # Sun flare (thêm ánh sáng chói - lens flare)
            # Use case: Simulate direct sunlight, lens flare
            # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.15),
            
            # Random shadow (thêm bóng tối)
            # Use case: Simulate shadows from objects, changing lighting
            # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            
            # ==================== QUALITY DEGRADATION ====================
            # Downscale (giảm resolution rồi scale lại - làm mất chi tiết)
            # Use case: Simulate low-resolution cameras, poor quality images
            # A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),
            
            # JPEG compression (nén JPEG - artifacts)
            # Use case: Simulate compressed images, JPEG artifacts
            # A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
            
            # Sharpen (làm sắc nét)
            # Use case: Simulate sharpening filters, enhance edges
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            
            # Emboss (hiệu ứng nổi - edge detection like)
            # Use case: Enhance edges, simulate emboss effect
            # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
            
            # ==================== DROPOUT & OCCLUSION ====================
            # Coarse Dropout (xóa các vùng hình chữ nhật lớn)
            # Use case: Simulate occlusions, missing parts, force model look at context
            # A.CoarseDropout(
            #     max_holes=5,        # Tối đa 5 vùng bị xóa
            #     max_height=60,      # Cao tối đa 60px
            #     max_width=60,       # Rộng tối đa 60px
            #     min_holes=1,
            #     min_height=20,
            #     min_width=20,
            #     fill_value=0,       # Fill bằng màu đen
            #     p=0.3
            # ),
            
            # Grid Dropout (xóa theo dạng lưới)
            # Use case: Simulate grid-like occlusions, mesh overlays
            # A.GridDropout(ratio=0.3, unit_size_min=10, unit_size_max=30, p=0.2),
            
            # Channel Dropout (xóa ngẫu nhiên 1 channel màu)
            # Use case: Force model không phụ thuộc vào 1 channel cụ thể
            # A.ChannelDropout(channel_drop_range=(1, 1), fill_value=128, p=0.2),
            
            # ==================== PIXEL-LEVEL EFFECTS ====================
            # Pixel Dropout (xóa ngẫu nhiên từng pixels riêng lẻ)
            # Use case: Simulate dead pixels, salt-and-pepper noise
            # A.PixelDropout(dropout_prob=0.01, per_channel=False, p=0.2),
            
            # Spatter (hiệu ứng văng bùn/nước)
            # Use case: Simulate mud splatter, water drops on lens
            # A.Spatter(mean=0.65, std=0.3, gauss_sigma=2, cutout_threshold=0.68, p=0.2),
            
            # To Grayscale (chuyển sang ảnh xám)
            # Use case: Force model học features không phụ thuộc màu sắc
            # A.ToGray(p=0.1),
            
            # To Sepia (hiệu ứng sepia - ảnh cổ)
            # Use case: Color augmentation, vintage effect
            # A.ToSepia(p=0.1),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_area=100,  # Minimum bbox area in pixels
            min_visibility=0.2,  # Minimum visibility after transform
            label_fields=['class_labels']
        ))
    
    def load_yolo_annotations(self, label_path):
        """Load YOLO format annotations (class x_center y_center width height)"""
        if not label_path.exists():
            return [], []
        
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_annotations(self, label_path, bboxes, class_labels):
        """Save YOLO format annotations"""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                # Ensure class_id is integer
                class_id = int(class_id)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_image(self, image, bboxes, class_labels):
        """Apply augmentation to image and bboxes"""
        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            print(f"Warning: Augmentation failed - {e}")
            return image, bboxes, class_labels
    
    def augment_dataset(self, split='train'):
        """Augment entire dataset for a split (train/val/test)"""
        print(f"\n{'='*80}")
        print(f"Augmenting {split} split")
        print(f"{'='*80}")
        
        # Paths
        img_dir = self.input_dir / 'images' / split
        label_dir = self.input_dir / 'labels' / split
        
        out_img_dir = self.output_dir / 'images' / split
        out_label_dir = self.output_dir / 'labels' / split
        
        if not img_dir.exists():
            print(f"⚠️  {split} split not found at {img_dir}")
            return
        
        # Create output dirs
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        if len(img_files) == 0:
            print(f"⚠️  No images found in {img_dir}")
            return
        
        print(f"Found {len(img_files)} images")
        print(f"Augmentation factor: {self.augment_factor}x")
        print(f"Output: {len(img_files) * (self.augment_factor + 1)} images total\n")
        
        augmented_count = 0
        failed_count = 0
        
        # Process each image
        for img_path in tqdm(img_files, desc=f"Processing {split}"):
            # Get corresponding label file
            label_path = label_dir / f"{img_path.stem}.txt"
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Failed to load: {img_path}")
                failed_count += 1
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load annotations
            bboxes, class_labels = self.load_yolo_annotations(label_path)
            
            # Copy original
            out_img_path = out_img_dir / img_path.name
            out_label_path = out_label_dir / label_path.name
            
            shutil.copy(img_path, out_img_path)
            if label_path.exists():
                shutil.copy(label_path, out_label_path)
            
            # Generate augmented versions
            for aug_idx in range(self.augment_factor):
                try:
                    # Augment
                    aug_image, aug_bboxes, aug_labels = self.augment_image(
                        image, bboxes, class_labels
                    )
                    
                    # Save augmented image
                    aug_img_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                    aug_img_path = out_img_dir / aug_img_name
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), aug_image_bgr)
                    
                    # Save augmented annotations
                    aug_label_path = out_label_dir / f"{img_path.stem}_aug{aug_idx}.txt"
                    self.save_yolo_annotations(aug_label_path, aug_bboxes, aug_labels)
                    
                    augmented_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed augmentation for {img_path.name} (aug{aug_idx}): {e}")
                    failed_count += 1
        
        print(f"\n✅ {split} split complete!")
        print(f"   Original: {len(img_files)}")
        print(f"   Augmented: {augmented_count}")
        print(f"   Total: {len(img_files) + augmented_count}")
        if failed_count > 0:
            print(f"   ⚠️  Failed: {failed_count}")
    
    def update_yaml(self):
        """Update data.yaml with new paths"""
        yaml_path = self.input_dir / 'data.yaml'
        out_yaml_path = self.output_dir / 'data.yaml'
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Update paths
            for split in ['train', 'val', 'test']:
                if split in data or f'{split}_path' in data:
                    old_path = data.get(split, data.get(f'{split}_path', ''))
                    if old_path:
                        new_path = str(self.output_dir / 'images' / split)
                        data[split] = new_path
            
            # Save updated yaml
            with open(out_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            print(f"\n✅ Updated data.yaml saved to {out_yaml_path}")
        else:
            print(f"\n⚠️  data.yaml not found at {yaml_path}")
    
    def copy_classes_file(self):
        """Copy classes.txt if exists"""
        classes_path = self.input_dir / 'classes.txt'
        if classes_path.exists():
            shutil.copy(classes_path, self.output_dir / 'classes.txt')
            print(f"✅ Copied classes.txt")
    
    def run(self, splits=['train']):
        """Run augmentation for specified splits"""
        print(f"\n{'='*80}")
        print(f"YOLO DATA AUGMENTATION")
        print(f"{'='*80}")
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Augmentation factor: {self.augment_factor}x")
        print(f"Splits: {splits}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Augment each split
        for split in splits:
            self.augment_dataset(split)
        
        # Copy/update config files
        self.update_yaml()
        self.copy_classes_file()
        
        print(f"\n{'='*80}")
        print(f"✅ AUGMENTATION COMPLETE!")
        print(f"{'='*80}")
        print(f"Augmented dataset: {self.output_dir}")
        print(f"\nTo train with augmented data:")
        print(f"  yolo train data={self.output_dir}/data.yaml ...")


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset')
    parser.add_argument('--input', type=str, default='custom_yolo_data',
                       help='Input YOLO dataset directory')
    parser.add_argument('--output', type=str, default='custom_yolo_data_augmented',
                       help='Output directory for augmented dataset')
    parser.add_argument('--augment-factor', type=int, default=10,
                       help='Number of augmented images per original image')
    parser.add_argument('--splits', nargs='+', default=['train'],
                       help='Splits to augment (train, val, test)')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import albumentations
    except ImportError:
        print("❌ albumentations not installed!")
        print("Install: pip install albumentations")
        return
    
    # Run augmentation
    augmenter = YOLODataAugmenter(
        input_dir=args.input,
        output_dir=args.output,
        augment_factor=args.augment_factor
    )
    
    augmenter.run(splits=args.splits)


if __name__ == '__main__':
    main()
