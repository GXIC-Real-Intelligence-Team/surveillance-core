import cv2
import subprocess
import argparse
import json
import requests
from gxic_rit.konan import faceapi
from gxic_rit.konan import image

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="debug mode, display frame", action="store_true")
parser.add_argument("-i", "--input", help="input rtmp stream", type=str, required=True)
parser.add_argument("-o", "--output", help="input rtmp stream", type=str, required=True)
parser.add_argument("-s", "--size", help="output video size, like 640x480", type=str, default="640x480")
args = parser.parse_args()

face_seq = 0

def request(ip, port, path, data=""):
    uri = "http://%s:%s/%s" % (ip, port, path)
    req = requests.post(uri, data=data)
    ret = json.loads(req.content)

    if args.debug:
        print(ret)

    return ret


def predict_face(peoples):
    data = []

    for people in peoples:
        item = {
            "seq": people['seq'],
            "eigen": people['eigen']
        }
        data.append(item)

    ret = request('127.0.0.1', 5000, "/predict", data=json.dumps(data))

    for item in ret:
        for people in peoples:
            if item['seq'] != people['seq']:
                continue

            if item['people']['confidence'] < 0.7:
                people['id'] = -1
                people['name'] = ''
                break

            people['id'] = item['people']['id']
            people['name'] = item['people']['name']


def mark_face(frame, peoples):
    for people in peoples:
        bb = people['bb']
        name = "%s#%d" % (people['name'], people['id'])

        scale = 0.2

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
        people['eigen'] = faceapi.getEigen(face)
        people['bb'] = bb

        face_seq += 1

    # call predict api here
    predict_face(peoples)
    return peoples


def main(args):
    try:
        ff_param = ["/usr/local/bin/ffmpeg",
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', args.size,  # size of one frame
                   '-pix_fmt', 'bgr24',
                   '-r', '24',  # frames per second
                   '-i', '-',  # The imput comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-f', 'flv',
                   args.output]

        cap = cv2.VideoCapture()
        cap.open(args.input)

        pipe = subprocess.Popen(ff_param, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        while (cap.isOpened()):
            ret, origin = cap.read()
            if ret == True:
                peoples = detect_face(origin)

                if len(peoples):
                    mark_face(origin, peoples)

                width = int(args.size.split('x')[0])
                height = int(args.size.split('x')[1])

                frame = cv2.resize(origin, (width, height))

                # sleep(0.5)

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


if __name__ == '__main__':
    main()