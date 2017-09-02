from bamos/openface
maintainer ruohan.chen<crhan123@gmail.com

add files/sources.list /etc/apt/sources.list
run apt-get update
run apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev freeglut3-dev mesa-common-dev  libgtkglext1 libgtkglext1-dev checkinstall yasm libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
run apt-get install libmysqlclient15-dev -y
add . /source
run pip install -r /source/requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
run pip install -e /source
workdir /source
