"""
colors.py
A color manager implemented with pure functions. 
Provides 0~1 gradient color calculation and retrieval of various solid colors.
The returned color format defaults to an (R, G, B) tuple (range 0~255).

Usage:
    import colors

    # --- Retrieve a Solid Color ---
    my_red = colors.get_color("red")            # Result: (255, 0, 0)
    my_hex_bg = colors.rgb_to_hex(my_red)       # Result: '#FF0000'

    # --- Get Gradient Colors (0 ~ 0.5 ~ 1.0) ---
    val1 = 0.25
    # At 0.25, it's halfway between Red and White
    c1 = colors.gradient_red_blue(val1)         # Result: (255, 127, 127)

    val2 = 0.75
    # At 0.75, it's halfway between White and Blue
    c2 = colors.gradient_red_blue(val2)         # Result: (127, 127, 255)

    # --- Use a Custom Split Boundary (e.g., split at 0.7) ---
    c3 = colors.gradient_orange_green(0.6, split_val=0.7) 

    # --- Test Abrupt Boundary Shift (continuous=False) ---
    # Here, 0.49 is light red, and 0.51 instantly shifts to light blue
    left_side = colors.gradient_red_blue(0.49, continuous=False)
    right_side = colors.gradient_red_blue(0.51, continuous=False)
"""

# ==========================================
# 1. Basic Solid Colors Definition & Retrieval
# ==========================================

_SOLID_COLORS = {
    # Basic colors
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    
    # Variant / Mixed colors
    "orange": (255, 128, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
}

def get_color(name, default_color=(0, 0, 0)):
    """
    External call: Get a solid color RGB tuple by color name.
    Example: get_color("red") -> (255, 0, 0)
    """
    return _SOLID_COLORS.get(name.lower().strip(), default_color)

def get_all_color_names():
    """External call: Get a list of all supported solid color names."""
    return list(_SOLID_COLORS.keys())


# ==========================================
# 2. Utilities: Calculation & Conversion
# ==========================================

def clamp(val, min_val=0.0, max_val=1.0):
    """Clamp a value between [min_val, max_val] to prevent overflow."""
    return max(min_val, min(val, max_val))

def lerp_color(color1, color2, t):
    """
    Linear interpolation between two RGB colors. 
    't' represents the progress from 0.0 to 1.0.
    """
    t = clamp(t)
    return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))

def rgb_to_hex(rgb):
    """
    External call: Convert an (R, G, B) tuple to standard web/UI hex format.
    Example: rgb_to_hex((255, 0, 0)) -> '#FF0000'
    """
    return "#{:02x}{:02x}{:02x}".format(clamp(rgb[0], 0, 255), 
                                        clamp(rgb[1], 0, 255), 
                                        clamp(rgb[2], 0, 255)).upper()


# ==========================================
# 3. Core Gradient Algorithm
# ==========================================

def get_split_gradient(val, start_color, left_mid_color, right_mid_color, end_color, split_val=0.5):
    """
    A generic two-stage gradient dispatch function. Uses split_val (default 0.5) as the boundary.
    0.0 ~ split_val: Color fades from start_color to left_mid_color
    split_val ~ 1.0: Color fades from right_mid_color to end_color
    """
    val = clamp(val)
    if val <= split_val:
        # Calculate left-side gradient progress (0.0 -> 1.0)
        t = val / split_val if split_val > 0 else 0
        return lerp_color(start_color, left_mid_color, t)
    else:
        # Calculate right-side gradient progress (0.0 -> 1.0)
        t = (val - split_val) / (1.0 - split_val) if split_val < 1.0 else 0
        return lerp_color(right_mid_color, end_color, t)


# ==========================================
# 4. Specific Gradient Functions
# ==========================================

def gradient_red_blue(val, split_val=0.5, continuous=True):
    """
    External call: Red to Blue two-stage gradient.
    :param val: A value between 0.0 and 1.0
    :param split_val: The boundary position, default is 0.5
    :param continuous: True for smooth transition via white (Red -> White -> Blue); 
                       False for an abrupt physical shift at the boundary (Red | Blue)
    """
    if continuous:
        return get_split_gradient(val, 
                                  start_color=(255, 0, 0),         # Pure Red
                                  left_mid_color=(255, 255, 255),  # White
                                  right_mid_color=(255, 255, 255), # White
                                  end_color=(0, 0, 255),           # Pure Blue
                                  split_val=split_val)
    else:
        return get_split_gradient(val, 
                                  start_color=(128, 0, 0),         # Dark Red
                                  left_mid_color=(255, 50, 50),    # Light Red
                                  right_mid_color=(50, 50, 255),   # Light Blue
                                  end_color=(0, 0, 128),           # Dark Blue
                                  split_val=split_val)

def gradient_orange_green(val, split_val=0.5, continuous=True):
    """
    External call: Orange to Green two-stage gradient 
    (0~0.5 Orange shades, 0.5~1 Green shades)
    """
    if continuous:
        return get_split_gradient(val, 
                                  start_color=(255, 128, 0),       # Orange
                                  left_mid_color=(255, 255, 255), 
                                  right_mid_color=(255, 255, 255), 
                                  end_color=(0, 150, 0),           # Green
                                  split_val=split_val)
    else:
        return get_split_gradient(val, 
                                  start_color=(150, 75, 0),        # Dark Orange
                                  left_mid_color=(255, 128, 0),    # Pure Orange
                                  right_mid_color=(0, 200, 0),     # Light Green
                                  end_color=(0, 100, 0),           # Dark Green
                                  split_val=split_val)

def gradient_yellow_purple(val, split_val=0.5, continuous=True):
    """
    External call: Yellow to Purple two-stage gradient 
    (0~0.5 Yellow shades, 0.5~1 Purple shades)
    """
    if continuous:
        return get_split_gradient(val, 
                                  start_color=(255, 215, 0),       # Gold / Yellow
                                  left_mid_color=(255, 255, 255), 
                                  right_mid_color=(255, 255, 255), 
                                  end_color=(128, 0, 128),         # Purple
                                  split_val=split_val)
    else:
        return get_split_gradient(val, 
                                  start_color=(150, 150, 0),       # Dark Yellow
                                  left_mid_color=(255, 255, 0),    # Pure Yellow
                                  right_mid_color=(180, 50, 180),  # Light Purple
                                  end_color=(75, 0, 75),           # Dark Purple
                                  split_val=split_val)