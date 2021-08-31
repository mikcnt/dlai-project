import numpy as np


def standardize_colors(img, distribution_mode="hard"):
    """Standardize enemy and ally ships colors for Procgen Plumber.
    This function will only work with 64x64x3 images, and its meant to be used
    with frames without background and using monochrome assets.
    Ally ships will be colored with white, enemy ships will be colored with red."""

    assert img.shape == (
        64,
        64,
        3,
    ), "`standardize_colors` only works with 64x64x3 images."

    # enemy ship model color will be in position (x = 57, y = 6) (bottom left angle area)
    enemy_ship_position = [57, 6]
    # color
    enemy_ship_color = img[enemy_ship_position[0], enemy_ship_position[1], :]

    # ally ship can move in the bottom center angle, in a small rectangle
    # at the right of the enemy ship model
    if distribution_mode == "hard":
        ally_ship_area = img[57:61, 13:, :]
    elif distribution_mode == "easy":
        ally_ship_area = img[55:61, 14:, :]

    # the ship is the only small non-black rectangle (!= 0)
    # to select the color, we just take any pixel color
    ally_ship_color = ally_ship_area[ally_ship_area != 0][:3]

    standardized_img = img.copy()

    # ally -> white
    standardized_img[np.where((standardized_img == ally_ship_color).all(axis=2))] = [
        255,
        255,
        255,
    ]
    # enemy -> red
    standardized_img[np.where((standardized_img == enemy_ship_color).all(axis=2))] = [
        255,
        0,
        0,
    ]

    #  change enemy ship model color
    if distribution_mode == "hard":
        standardized_img[51:, 0:13, :] = [255, 127, 63]
        standardized_img[55:60, 2:11, :] = [255, 0, 0]
    elif distribution_mode == "easy":
        standardized_img[51:, 0:13, :] = [255, 127, 63]
        standardized_img[53:62, 0:14, :] = [255, 0, 0]

    return standardized_img
