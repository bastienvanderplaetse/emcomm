from PIL import Image, ImageDraw
import numpy as np
import math

def draw_circle(draw, center_x, center_y, shape_size, shape_color, fill):
    radius = shape_size / 2
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=shape_color, width=3, fill=shape_color if fill else None)

def draw_square(draw, center_x, center_y, shape_size, shape_color, rotation_angle, fill):
    vertices = [
        (center_x - shape_size/2, center_y - shape_size/2),
        (center_x + shape_size/2, center_y - shape_size/2),
        (center_x + shape_size/2, center_y + shape_size/2),
        (center_x - shape_size/2, center_y + shape_size/2)
    ]
    rotated_vertices = [(x - center_x, y - center_y) for x, y in vertices]
    rotated_vertices = [(x * math.cos(math.radians(rotation_angle)) - y * math.sin(math.radians(rotation_angle)),
                     x * math.sin(math.radians(rotation_angle)) + y * math.cos(math.radians(rotation_angle)))
                    for x, y in rotated_vertices]
    rotated_vertices = [(x + center_x, y + center_y) for x, y in rotated_vertices]
    draw.polygon(rotated_vertices, outline=shape_color, width=3, fill=shape_color if fill else None)

def draw_cross(draw, center_x, center_y, shape_size, shape_color, rotation_angle):
    vertices = [
        (center_x - shape_size/2, center_y - shape_size/2),
        (center_x + shape_size/2, center_y - shape_size/2),
        (center_x + shape_size/2, center_y + shape_size/2),
        (center_x - shape_size/2, center_y + shape_size/2)
    ]
    rotated_vertices = [(x - center_x, y - center_y) for x, y in vertices]
    rotated_vertices = [(x * math.cos(math.radians(rotation_angle)) - y * math.sin(math.radians(rotation_angle)),
                     x * math.sin(math.radians(rotation_angle)) + y * math.cos(math.radians(rotation_angle)))
                    for x, y in rotated_vertices]
    rotated_vertices = [(x + center_x, y + center_y) for x, y in rotated_vertices]
    draw.line((rotated_vertices[0], rotated_vertices[2]), fill=shape_color,width=3)
    draw.line((rotated_vertices[1], rotated_vertices[3]), fill=shape_color,width=3)


def shift(center_x_1, center_y_1, center_x_2, center_y_2, size, shape_size_1, shape_size_2):
    radius_1 = shape_size_1 / 2 + 5
    radius_2 = shape_size_2 / 2 + 5
    if center_x_2 < radius_2:
        diff = radius_2 - center_x_2
        center_x_2 += diff
        center_x_1 += diff
        if center_x_1 > size-radius_1:
            center_x_1 = center_x_1 - (center_x_1-size+radius_1)
    elif center_x_2 > size-radius_2:
        diff = center_x_2 - size + radius_2
        center_x_2 -= diff 
        center_x_1 -= diff
        if center_x_1 < radius_1:
            center_x_1 = radius_1

    if center_y_2 < radius_2:
        diff = radius_2 - center_y_2
        center_y_2 += diff 
        center_y_1 += diff
        if center_x_1 > size-radius_1:
            center_x_1 = center_x_1 - (center_x_1-size+radius_1)

    return center_x_1, center_y_1, center_x_2, center_y_2

def generate_positions(relation, shape_size_1, shape_size_2, size):
    center_x_1 = np.random.randint(shape_size_1, size-shape_size_1 + 1)
    center_y_1 = np.random.randint(shape_size_1, size-shape_size_1 + 1)
    
    if relation == "right":
        shift_x = np.random.randint(50, 89)
        shift_y = np.random.randint(-5, 6)
    elif relation == "topright":
        shift_x = np.random.randint(50, 88)
        shift_y = np.random.randint(-88, -49)
    elif relation == "top":
        shift_x = np.random.randint(-5, 6)
        shift_y = np.random.randint(-88, -49)
    elif relation == "topleft":
        shift_x = np.random.randint(-88, -49)
        shift_y = np.random.randint(-88, -49)

    center_x_2 = center_x_1 + shift_x
    center_y_2 = center_y_1 + shift_y
    
    center_x_1, center_y_1, center_x_2, center_y_2 = shift(center_x_1, center_y_1, center_x_2, center_y_2, size, shape_size_1, shape_size_2)

    return center_x_1, center_y_1, center_x_2, center_y_2

def generate(shape1, shape2, relation, size=128):
    background_color = (0,0,0) # black background
    shape_color = (255, 255, 255) # white shape

    image = Image.new("RGB", (size, size), background_color)
    image_array = np.array(image)

    noise = np.random.normal(0.0, 16.0, image_array.shape).astype(int)
    noisy_background_color = (background_color + noise)%255
    
    image = Image.fromarray(noisy_background_color.astype(np.uint8))

    draw = ImageDraw.Draw(image)

    shape_size_1 = np.random.randint(28, 41)
    shape_size_2 = np.random.randint(28,41)
    shape_size_1 = 28
    shape_size_2 = 28
   
    rotation_angle_1 = np.random.randint(0,360)
    rotation_angle_2 = np.random.randint(0,360)

    center_x_1, center_y_1, center_x_2, center_y_2 = generate_positions(relation, shape_size_1, shape_size_2, size)

    if shape1 == "circle":
        draw_circle(draw, center_x_1, center_y_1, shape_size_1, shape_color, fill=False)
    elif shape1 == "circlefill":
        draw_circle(draw, center_x_1, center_y_1, shape_size_1, shape_color, fill=True)
    elif shape1 == "square":
        draw_square(draw, center_x_1, center_y_1, shape_size_1, shape_color, rotation_angle_1, fill=False)
    elif shape1 == "squarefill":
        draw_square(draw, center_x_1, center_y_1, shape_size_1, shape_color, rotation_angle_1, fill=True)
    elif shape1 == "cross":
        draw_cross(draw, center_x_1, center_y_1, shape_size_1, shape_color, rotation_angle_1)

    if shape2 == "circle":
        draw_circle(draw, center_x_2, center_y_2, shape_size_2, shape_color, fill=False)
    elif shape2 == "circlefill":
        draw_circle(draw, center_x_2, center_y_2, shape_size_2, shape_color, fill=True)
    elif shape2 == "square":
        draw_square(draw, center_x_2, center_y_2, shape_size_2, shape_color, rotation_angle_2, fill=False)
    elif shape2 == "squarefill":
        draw_square(draw, center_x_2, center_y_2, shape_size_2, shape_color, rotation_angle_2, fill=True)
    elif shape2 == "cross":
        draw_cross(draw, center_x_2, center_y_2, shape_size_2, shape_color, rotation_angle_2)
  
    # image.save("image_with_rectangle.png")
    return image