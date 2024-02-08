import cv2
import numpy as np

def get_equalised(face_resized):
    face_img = np.zeros_like(face_resized)
    h, w = face_resized.shape
    left_half = face_resized[:, :w // 2]
    right_half = face_resized[:, w // 2:]
    mid_x = w // 2
    # Apply histogram equalization separately to the left and right halves
    left_side = cv2.equalizeHist(left_half)
    right_side = cv2.equalizeHist(right_half)
    whole_face = cv2.equalizeHist(face_resized)
    for y in range(h):
        for x in range(w):
            if x < w // 4:
                # Left 25%: just use the left face.
                v = left_side[y, x]
            elif x < w * 2 // 4:
                # Mid-left 25%: blend the left face & whole face.
                lv = left_side[y, x]
                wv = whole_face[y, x]
                # Blend more of the whole face as it moves further right along the face.
                f = (x - w // 4) / (w / 4)
                v = int((1.0 - f) * lv + f * wv)
            elif x < w * 3 // 4:
                # Mid-right 25%: blend right face & whole face.
                rv = right_side[y, x - mid_x]
                wv = whole_face[y, x]
                # Blend more of the right-side face as it moves further right along the face.
                f = (x - w * 2 // 4) / (w / 4)
                v = int((1.0 - f) * wv + f * rv)
            else:
                # Right 25%: just use the right face.
                v = right_side[y, x - mid_x]

            face_img[y, x] = v
    return face_img

def get_ellipse(filtered):
    mask = np.ones_like(filtered, dtype=np.uint8) * 255

    # Define ellipse parameters
    dh, dw = filtered.shape
    face_center = (int(dw * 0.5), int(dh * 0.4))
    ellipse_size = (int(dw * 0.5), int(dh * 0.8))
    
    # Draw a black-filled ellipse in the middle of the mask
    cv2.ellipse(mask, face_center, ellipse_size, 0, 0, 360, 0, thickness=cv2.FILLED)

    # Apply the elliptical mask on the face, to remove corners.
    # Sets corners to gray, without touching the inner face.
    filtered[np.where(mask == 255)] = 128
    return filtered