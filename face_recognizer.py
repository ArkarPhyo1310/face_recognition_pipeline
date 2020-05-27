import joblib
import cv2
import numpy as np
from PIL import Image
from detect_recognize_pipeline import preprocessing, FaceFeaturesExtractor
from util import draw_bb_on_img, draw_fps, draw_frame_count
import argparse
import os
from timeit import default_timer as timer


def get_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="webcam",
                    choices=['webcam', 'video', 'image'],
                    help="Choose one to test from ['webcam', 'video', 'image']")
    ap.add_argument("--path", default=0,
                    help="Path to video or image file")
    ap.add_argument("--model", default="model/face_recogniser.pkl",
                    help="Path to the Face Recognizer PKL file")
    ap.add_argument("--save", action='store_true',
                    help="Save the output [webcam, video, image].")
    return ap.parse_args()


def save_video(frame_array, width, height):
    # save the video
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    output_video = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(width), int(height)))
    for frame in frame_array:
        output_video.write(frame)


def main():
    args = get_arg()
    face_recogniser = joblib.load(args.model)
    feature_extractor = FaceFeaturesExtractor()
    preprocess = preprocessing.ExifOrientationNormalize()
    frame_array = []

    if args.type == 'webcam':
        webcam = cv2.VideoCapture(args.path)
        vidw = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        vidh = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        frame_count = 1

        while True:
            # Capture frame-by-frame
            ret, frame = webcam.read()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = face_recogniser(feature_extractor, preprocess(img))
            if faces is not None:
                draw_bb_on_img(faces, img)

            # Calculate FPS
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # draw fps and frame count
            draw_fps(img, fps, vidw)
            draw_frame_count(img, str(frame_count))

            # Display the resulting frame
            cv2.imshow('Recognizing Faces', cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
            frame_array.append(np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1

        if args.save:
            save_video(frame_array, vidw, vidh)
        # When everything done, release the capture
        webcam.release()
        cv2.destroyAllWindows()

    elif args.type == "video":
        if not str(args.path).endswith(('.mp4', '.avi')):
            print("[+] Unknown video file detected...")
            return
        video = cv2.VideoCapture(args.path)
        vidw = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        vidh = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        frame_count = 1
        while True:
            # Capture frame-by-frame
            ret, frame = video.read()
            if not ret:
                print("[+] finish reading video file...")
                break
            frame = cv2.flip(frame, 1)

            img = Image.fromarray(frame)
            img = img.convert('RGB')
            faces = face_recogniser(preprocess(img))
            if faces is not None:
                draw_bb_on_img(faces, img)

            # Calculate FPS
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # draw fps and frame count
            draw_fps(img, fps, vidw)
            draw_frame_count(img, str(frame_count))

            # Display the resulting frame
            cv2.imshow('video', np.array(img))
            frame_array.append(np.array(img))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # save the output
        if args.save:
            save_video(frame_array, vidw, vidh)

        # When everything done, release the capture
        video.release()
        cv2.destroyAllWindows()

    elif args.type == "image":
        img = Image.open(args.path)
        img = img.convert('RGB')
        faces = face_recogniser(preprocess(img))
        if faces is not None:
            draw_bb_on_img(faces, img)
        cv2.imshow(os.path.basename(args.path), cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        if args.save:
            cv2.imwrite('output.png', cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
