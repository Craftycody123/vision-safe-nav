def is_dangerous(box, threshold=50000):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    return area > threshold