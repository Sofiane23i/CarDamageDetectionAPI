import json
from collections import defaultdict
import os

def convert_coco_to_vgg(coco_json_path, output_json_path, image_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Create lookup tables
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image
    image_to_annotations = defaultdict(list)
    for ann in coco['annotations']:
        image_to_annotations[ann['image_id']].append(ann)

    vgg = {}

    for image_id, anns in image_to_annotations.items():
        image_info = images[image_id]
        filename = image_info['file_name']
        filepath = os.path.join(image_dir, filename)
        size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        regions = []
        for ann in anns:
            if not ann.get('segmentation'):
                continue

            # Assume polygon (not RLE)
            seg = ann['segmentation'][0]  # single polygon
            xs = seg[::2]
            ys = seg[1::2]

            regions.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": xs,
                    "all_points_y": ys
                },
                "region_attributes": {
                    "label": categories[ann['category_id']]
                }
            })

        vgg_key = f"{filename}{size}"
        vgg[vgg_key] = {
            "filename": filename,
            "size": size,
            "regions": regions
        }

    with open(output_json_path, "w") as f:
        json.dump(vgg, f, indent=2)

    print(f"✅ Converted to VGG format and saved to {output_json_path}")
