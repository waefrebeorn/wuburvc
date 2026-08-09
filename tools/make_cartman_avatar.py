"""Generate a Cartman-style avatar image for the video."""
from PIL import Image, ImageDraw
import numpy as np

img = Image.new('RGB', (512, 512), (255, 230, 170))
draw = ImageDraw.Draw(img)

cx, cy = 256, 256
r = 180
draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 228, 0))  # yellow face

# Red cap
draw.polygon([(cx-120, cy-150), (cx+120, cy-150), (cx+120, cy-50), (cx-120, cy-50)], fill=(255, 0, 0))
draw.polygon([(cx-40, cy-230), (cx+40, cy-230), (cx+80, cy-50), (cx-80, cy-50)], fill=(255, 0, 0))
draw.rectangle([cx-10, cy-220, cx+10, cy-50], fill=(255, 255, 255))

# Eyes
eye_r = 15
draw.ellipse([cx-50-eye_r, cy-30-eye_r, cx-50+eye_r, cy-30+eye_r], fill=(0,0,0))
draw.ellipse([cx+50-eye_r, cy-30-eye_r, cx+50+eye_r, cy-30+eye_r], fill=(0,0,0))

# Smile
draw.arc([cx-60, cy+20, cx+60, cy+100], start=20, end=160, fill=(200, 30, 30), width=8)

# Pink shirt collar
draw.polygon([(cx-70, cy+180), (cx+70, cy+180), (cx+60, cy+280), (cx-60, cy+280)], fill=(255, 180, 200))

img.save('outputs/cartman_avatar.png')
print(f"Saved Cartman avatar to outputs/cartman_avatar.png")
