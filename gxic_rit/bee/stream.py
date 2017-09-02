import cv2
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="debug mode, display frame", action="store_true")
parser.add_argument("-i", "--input", help="input rtmp stream", type=str, required=True)
parser.add_argument("-o", "--output", help="input rtmp stream", type=str, required=True)
parser.add_argument("-s", "--size", help="output video size, like 640x480", type=str, default="640x480")

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
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(frame, (640, 480))

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
    main(parser.parse_args())