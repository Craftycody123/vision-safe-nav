def get_direction(box, frame_width):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2

    if center_x < frame_width / 3:
        return "left"
    elif center_x > 2 * frame_width / 3:
        return "right"
    else:
        return "ahead"