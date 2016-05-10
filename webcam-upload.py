import numpy as np
import requests
import cv2

def grab_img(ramp_frames=0, cam_port=0):
    """Grab a single frame from the webcam.

    Parameters:
    -----------
    ramp_frames: the number of frames to discard while cam adjusts to light level
    cam_port: the webcam's device index

    Returns:
    --------
    frame: a single RBG image
    """
    cap = cv2.VideoCapture(cam_port)

    # get past the ramp_frames, if any
    for _ in range(ramp_frames):
        cap.read()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frame

def upload_img(img, temp_img_name='tmp.jpg', panda_api_url='http://54.210.9.61/panda_app/'):
    # save the image to a file
    cv2.imwrite(temp_img_name, img)

    with open(temp_img_name) as f:
        response = requests.post(panda_api_url, files={"file":f})

    return response

if __name__ == '__main__':
    cam_img = grab_img(ramp_frames=12)
    response = upload_img(cam_img)
    print response.text
