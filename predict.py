#----------------------------------------------------#
#   Single image, camera, FPS test in one file. Use mode to switch.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   Modify class colors in __init__ function
    #-------------------------------------------------------------------------#
    unet = Unet()
    #----------------------------------------------------------------------------------------------------------#
    #   mode: 'predict'=single image, 'video'=camera/video, 'fps'=test fps, 'dir_predict'=batch folder, 'export_onnx'=export onnx
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"   ### Image detection
    # mode = "video"     ## Video detection
    # mode = "video"
    #-------------------------------------------------------------------------#
    #   count: count pixels and ratio. name_classes: class names.
    #   Valid when mode='predict'
    #-------------------------------------------------------------------------#
    count           = False
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes = ["_background_", "person", "car", "motorbike", "dustbin", "chair", "fire_hydrant", "tricycle", "bicycle","stone"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path: video path, 0=camera. video_save_path: save path, ""=no save. video_fps: FPS.
    #   Valid when mode='video'. Save on ctrl+c or last frame.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval: fps test count. fps_image_path: test image.
    #   Valid when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path: image folder. dir_save_path: save folder.
    #   Valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify: use Simplify onnx. onnx_save_path: save path.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        predict.py notes:
        1. Batch prediction: use os.listdir() + Image.open(). See get_miou_prediction.py.
        2. Save: r_image.save("img.jpg").
        3. No blend: set blend=False.
        4. Get regions: see detect_image, use prediction to draw and get class regions.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()


    ## Below is camera detection
    elif mode == "video_camera":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera (video). Check if camera is installed (video path correct).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read one frame
            ref, frame = capture.read()
            if not ref:
                break
            # Format conversion, BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Detect
            frame = np.array(unet.detect_image(frame))
            # RGB to BGR for opencv display
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)


    ## Below is video file detection from disk
    elif mode == "video":

        while True:
            video = input("Input video filename:")
            try:
                capture = cv2.VideoCapture(video)
            except:
                print("Open Error! Try again!")
                continue
            else:
        # capture = cv2.VideoCapture(video_path)
                if video_save_path != "":
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

                ref, frame = capture.read()
                if not ref:
                    raise ValueError("Failed to read camera (video). Check if camera is installed (video path correct).")

                fps = 0.0
                while (True):
                    t1 = time.time()
                    # Read one frame
                    ref, frame = capture.read()
                    if not ref:
                        break
                    # Format conversion, BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to Image
                    frame = Image.fromarray(np.uint8(frame))
                    # Detect
                    frame = np.array(unet.detect_image(frame))
                    # RGB to BGR for opencv display
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    fps = (fps + (1. / (time.time() - t1))) / 2
                    print("fps= %.2f" % (fps))
                    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("video", frame)
                    c = cv2.waitKey(1) & 0xff
                    if video_save_path != "":
                        out.write(frame)

                    if c == 27:
                        capture.release()
                        break
                print("Video Detection Done!")
                capture.release()
                if video_save_path != "":
                    print("Save processed video to path :" + video_save_path)
                    out.release()
                cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")




