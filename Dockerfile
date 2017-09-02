from bamos/openface
maintainer ruohan.chen<crhan123@gmail.com

run apt-get update; apt-get install libmysqlclient15-dev -y
add . /source
run pip install -r /source/requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
run pip install -e /source
workdir /source
