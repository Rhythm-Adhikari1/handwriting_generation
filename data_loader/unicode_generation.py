import os
import re
from PIL import Image
import numpy as np
import freetype
import uharfbuzz as hb

image_width, image_height = 16, 16  # Slightly larger for better visibility
background_color = 255  # white
font_path = 'data_loader/NotoSansDevanagari-Regular.ttf'  

def generate_syllable_image(syllable_text):
    # --- Load font bytes for HarfBuzz ---
    with open(font_path, "rb") as f:
        fontdata = f.read()
    hb_face = hb.Face(fontdata)
    hb_font = hb.Font(hb_face)

    # --- Shape text ---
    buf = hb.Buffer()
    buf.add_str(syllable_text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)
    infos = buf.glyph_infos
    positions = buf.glyph_positions

    # --- Load FreeType for rasterization ---
    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, 15 ) ##

    # --- Compute bounding box ---
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    x_cursor = 0
    for info, pos in zip(infos, positions):
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bitmap = face.glyph.bitmap
        top = face.glyph.bitmap_top
        left = face.glyph.bitmap_left

        x0 = x_cursor + left
        y0 = -top
        x1 = x0 + bitmap.width
        y1 = y0 + bitmap.rows

        x_min = min(x_min, x0)
        y_min = min(y_min, y0)
        x_max = max(x_max, x1)
        y_max = max(y_max, y1)

        x_cursor += pos.x_advance / 64

    text_width = x_max - x_min
    text_height = y_max - y_min

    # --- Create image ---
    img = Image.new("L", (image_width, image_height), background_color)

    # --- Render glyphs centered ---
    x_cursor = (image_width - text_width) / 2 - x_min
    y_cursor = (image_height - text_height) / 2 - y_min

    # Define upper and lower matras for manual adjustment
    upper_symbols = ['े', 'ै']
    
    lower_symbols = ['्', 'ु', 'ू', 'ृ', 'ॄ']

    for ch, info, pos in zip(syllable_text, infos, positions):
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bitmap = face.glyph.bitmap
        top = face.glyph.bitmap_top
        left = face.glyph.bitmap_left

        # --- Apply manual offset correction ---
        y_offset_adjust = 0
        if ch in upper_symbols:
            y_offset_adjust = -2   # shift slightly upward
        elif ch in lower_symbols:
            y_offset_adjust = +1   # shift slightly downward

        if bitmap.buffer:
            glyph_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)
            glyph_array = 255 - glyph_array  # invert to black text
            glyph_img = Image.fromarray(glyph_array, mode="L")

            img.paste(
                glyph_img,
                (int(x_cursor + left), int(y_cursor - top + y_offset_adjust))
            )

        x_cursor += pos.x_advance / 64

    
    return img

