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


def draw_bbox(img, bbox,classifier_result ,color=[0, 0, 255], bbox_type='xyxy'):
    """
        bbox_type: ['xyxy', 'xywh']
    """
    if bbox_type == 'xywh':
        cx, cy, w, h = bbox
        c1, c2 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
    elif bbox_type == 'xyxy':
        xmin, ymin, xmax, ymax = bbox[0]
        c1, c2 = (int(xmin), int(ymin)), (int(xmax),int(ymax))
    elif bbox_type == 'x1y1wh':
        xmin, ymin, w, h = bbox.astype(int)
        c1, c2 = (xmin, ymin), (xmin + w, ymin + h)
    print(classifier_result[0])
    w, h = cv2.getTextSize(classifier_result[0], 0, fontScale=10, thickness=2)[0] 
    outside = c1[1] - h >= 3
    cv2.rectangle(img, c1, c2, color, 2, cv2.LINE_AA)
    cv2.putText(img,classifier_result[0], (c1[0], c1[1] - 2 if outside else c1[1] + h + 2),
                            0,
                            3,
                            color,
                            thickness=2,
                            lineType=cv2.LINE_AA)
    return img
