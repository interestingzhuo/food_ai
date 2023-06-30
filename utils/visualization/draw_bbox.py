import cv2
import imageio
from tqdm import tqdm


def draw_body_bbox(video, filename, results):
    writer = imageio.get_writer(filename, fps=30, macro_block_size=1)
    for image, result in tqdm(zip(video, results), desc='body bbox visualiztion', total=len(results)):
        image = draw_bbox(image, result[0], bbox_type='x1y1wh')
        writer.append_data(image[..., ::-1])
    writer.close()


def draw_hand_bbox(video, filename, results):
    writer = imageio.get_writer(filename, fps=30, macro_block_size=1)
    for image, result in tqdm(zip(video, results), desc='hand bbox visualiztion', total=len(results)):
        if 'left_hand' in result:
            image = draw_bbox(image, result['left_hand'][0], [0, 255, 0], bbox_type='x1y1wh')
        if 'right_hand' in result:
            image = draw_bbox(image, result['right_hand'][0], [255, 0, 255], bbox_type='x1y1wh')
        writer.append_data(image[..., ::-1])
    writer.close()


def draw_bbox(img, bbox, color=[0, 0, 255], bbox_type='xyxy'):
    """
        bbox_type: ['xyxy', 'xywh']
    """
    if bbox_type == 'xywh':
        cx, cy, w, h = bbox
        c1, c2 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
    elif bbox_type == 'xyxy':
        xmin, ymin, xmax, ymax = bbox.astype(int)
        c1, c2 = (xmin, ymin), (xmax, ymax)
    elif bbox_type == 'x1y1wh':
        xmin, ymin, w, h = bbox.astype(int)
        c1, c2 = (xmin, ymin), (xmin + w, ymin + h)

    cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

    return img
