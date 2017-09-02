from bamos/openface
maintainer ruohan.chen<crhan123@gmail.com

add files/sources.list /etc/apt/sources.list
run apt-get update
run apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev freeglut3-dev mesa-common-dev  libgtkglext1 libgtkglext1-dev checkinstall yasm libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
run apt-get install libmysqlclient15-dev -y
add requirements.txt /tmp/requirements.txt
run pip install -r /tmp/requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
run add-apt-repository ppa:mc3man/trusty-media -y
run apt-get update
run apt-get install ffmpeg -y
run apt-get install vim -y

add . /source
run pip install -e /source
add files/cv2.so /usr/local/lib/python2.7/dist-packages/cv2/cv2.so
workdir /source
