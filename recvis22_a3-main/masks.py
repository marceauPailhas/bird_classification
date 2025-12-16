import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator , SamPredictor
import os

# --- Load the image ---
# image_path = "./bird_dataset/train_images/004.Groove_billed_Ani/Groove_Billed_Ani_0002_1670.jpg"  # path to your image
# image_folder_path = "bird_dataset/train_images/004.Groove_billed_Ani"
def compute_bird_mask(image_path, overlay_path, mask_path):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # --- Load SAM model ---
  sam_checkpoint = "./recvis22_a3-main/sam_weights/sam_vit_h_4b8939.pth"  # download from Meta's SAM repo if not present
  model_type = "vit_h"

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

  predictor = SamPredictor(sam)
  predictor.set_image(image)
  height, width = image.shape[:2]
  # --- Interactive click selection ---
  clicked_point = []

  def click_event(event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:
          clicked_point.append((x, y))
          print(f"Clicked at: ({x}, {y})")
          cv2.destroyAllWindows()  # close window after click

  # Show image and wait for user click
  temp_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  cv2.namedWindow("Click on the bird")
  cv2.setMouseCallback("Click on the bird", click_event)

  while True:
      cv2.imshow("Click on the bird", temp_img)
      key = cv2.waitKey(20)  # refresh window
      if clicked_point:
          break
      if key == 27:   # ESC to cancel
          raise RuntimeError("Cancelled.")


  if not clicked_point:
      raise RuntimeError("No point selected! Please click on the bird in the image window.")

  # --- Use clicked point for SAM prediction ---
  input_point = np.array([clicked_point[0]])  # (x, y)
  input_label = np.array([1])  # foreground
  print(f"Using input point: {input_point}")


  masks, _, _ = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=True
      )

  # mask_generator = SamAutomaticMaskGenerator(sam)

  # # --- Generate all masks ---
  # masks = mask_generator.generate(image)

  # --- Optional: filter masks that look like birds ---
  # (This step depends on your use case; SAM does not classify by category.)
  # For demonstration, we assume you have external logic to pick bird-like masks.
  # Here, we’ll just combine all masks into one binary mask.

  binary_mask = np.zeros((height, width), dtype=np.uint8)

  # for mask_data in masks:
  #     m = mask_data["segmentation"]
  #     binary_mask[m] = 1
  for m in masks:
    binary_mask[m] = 1

  # --- Save or visualize the binary mask ---
  cv2.imwrite(mask_path, binary_mask * 255)
  print("Saved mask to:", mask_path)

  # Optional: visualize overlay
  overlay = (image * 0.7 + np.stack([binary_mask * 255] * 3, axis=-1) * 0.3).astype(np.uint8)
  cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
  print("Saved overlay to:", overlay_path)

folder = "bird_dataset/train_images"   # your folder

png_files = []

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(".jpg") and ("overlays" not in root.lower()) and ("masks" not in root.lower()) :
            if os.path.exists (os.path.join(root , "masks", file) ) :
                continue
            full_path = os.path.join(root, file)
            png_files.append(full_path)

print("Found PNG files:")
for path in png_files:
    dir_path = os.path.dirname(path)     
    file_name = os.path.basename(path) 
    print(path)
    overlay_dir = os.path.join(dir_path, "overlays")
    mask_dir = os.path.join(dir_path, "masks")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    compute_bird_mask(path, os.path.join(overlay_dir, file_name), os.path.join(mask_dir, file_name))
    


#compute_bird_mask(image_path, "bird_overlay.png", "bird_mask.png")