import cv2
import subprocess
import argparse
import json
import traceback
import requests
from gxic_rit.konan import faceapi
from gxic_rit.konan import image
from gxic_rit.bee.webcamvideostream import WebcamVideoStream

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--debug", help="debug mode, display frame", action="store_true")
parser.add_argument("-i", "--input", help="input rtmp stream",
                    type=str, required=True)
parser.add_argument(
    "-o", "--output", help="input rtmp stream", type=str, required=True)
parser.add_argument(
    "-s", "--size", help="output video size, like 640x480", type=str, default="640x480")
parser.add_argument('-r', '--rate', default=30)
args = parser.parse_args()

face_seq = 0
CONFIDENCE = 0.2

assert 'FFMPEG:                      YES' in cv2.getBuildInformation()


def predict_face(peoples):
    data = []

    for people in peoples:
        item = {
            "seq": people['seq'],
            "eigen": people['eigen']
        }
        data.append(item)

    resp = requests.post("http://gxic.crhan.com:5000/predict", json=data)
    ret = resp.json()

    for item in ret:
        for people in peoples:
            if item['seq'] != people['seq']:
                continue

            print("get {} confidence {}".format(item['people']['name'],
                                                item['people']['confidence']))
            if item['people']['confidence'] < CONFIDENCE:
                people['id'] = -1
                people['name'] = ''
                break

            people['id'] = item['people']['id']
            people['name'] = item['people']['name']
            people['confidence'] = item['people']['confidence']


def mark_face(frame, peoples):
    for people in peoples:
        bb = people['bb']
        name = "%s#%d#%f" % (people['name'], people['id'], people['confidence'])

        scale = 1

        image.printFaceBox(frame, scale, bb)
        image.printName(frame, name, scale, bb)


def detect_face(frame):
    global face_seq

    bbs = faceapi.allFaceBoundingBoxes(frame)
    peoples = []
    for bb in bbs:
        landmarks = faceapi.findLandmarks(frame, bb)
        face = faceapi.align(frame, bb, landmarks=landmarks)

        if face is None:
            continue

        people = {}
        people['seq'] = face_seq
        people['eigen'] = faceapi.getEigen(face).tolist()
        people['bb'] = bb
        peoples.append(people)

        face_seq += 1

    # call predict api here
    predict_face(peoples)
    return peoples


def main():
    try:
        ff_param = ["/usr/bin/ffmpeg",
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', args.size,  # size of one frame
                    '-pix_fmt', 'bgr24',
                    '-r', '24',  # frames per second
                    '-i', '-',  # The imput comes from a pipe
                    '-an',  # Tells FFMPEG not to expect any audio
                    '-g', '1',  # make every frame a keyframe
                    '-r', str(args.rate),
                    '-f', 'flv',
                    '-vcodec', 'libx264',
                    args.output]

        #if args.input == "local":
        #    cap = cv2.VideoCapture(0)
        #else:
        #    cap = cv2.VideoCapture(args.input)

        cap = WebcamVideoStream(src=args.input)
        cap.start()

        pipe = subprocess.Popen(
            ff_param, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        cnt = 0
        while (cap.isOpened()):
            ret, origin = cap.read()

            if ret is True:
                width = int(args.size.split('x')[0])
                height = int(args.size.split('x')[1])
                frame = cv2.resize(origin, (width, height))

                peoples = detect_face(frame)
                print('find {} people in frame'.format(len(peoples)))

                if peoples:
                    print("mark_face for {} peoples".format(len(peoples)))
                    mark_face(frame, peoples)

                pipe.stdin.write(frame)

                if args.debug:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)

        # Release everything if job is finished
        cap.release()

        if args.debug:
            cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
