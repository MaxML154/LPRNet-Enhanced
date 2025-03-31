from PIL import ImageDraw

# Monkey patch for Pillow/PIL >= 10.0.0 where textsize is deprecated
if not hasattr(ImageDraw.ImageDraw, 'textsize'):
    def textsize(self, text, font=None, spacing=4, direction=None, features=None, language=None, stroke_width=0):
        """
        Compatibility function for PIL >= 10.0.0 that reimplements textsize using textbbox
        """
        bbox = self.textbbox((0, 0), text, font, spacing, direction, features, language, stroke_width)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    ImageDraw.ImageDraw.textsize = textsize

def apply_pillow_patches():
    """Call this function to apply Pillow compatibility patches"""
    # Currently just applies the textsize patch
    pass 