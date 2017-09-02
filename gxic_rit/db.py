# -*- coding: utf-8 -*-

import json

from sqlalchemy import create_engine, Table, MetaData, select, Column, Integer, String

from gxic_rit.konan.people import People


db_url = "mysql://gxic:15NCp42Zz7hOy7gGX@rm-bp1i84r22ct5689kh.mysql.rds.aliyuncs.com:3306/gxic"

meta = MetaData()
Visitors = Table('visitors', meta,
                 Column('id', Integer, primary_key=True),
                 Column('is_leave', Integer),
                 Column('name', String),
                 Column('eigens', String)
                 )

Alarm = Table('alarms', meta,
              Column('id', Integer, primary_key=True),
              Column('visitor_id', Integer),
              Column('camera_id', Integer),
              )

engines = {}


def get_all_visitors(eng=None):
    """
    yield 一系列 People 对象
    """
    if eng is None:
        eng = get_engine()

    if meta.bind is None:
        eng = create_engine(db_url)
        meta.bind = eng

    with eng.connect() as conn:
        s = select([Visitors.c.id, Visitors.c.name, Visitors.c.eigens]).where(Visitors.c.is_leave == 0)
        for row in conn.execute(s):
            if not row['eigens']:
                continue
            yield People(pid=row['id'], name=row['name'],
                         eigens=json.loads(row['eigens']))


def get_engine():
    if db_url not in engines:
        engines[db_url] = create_engine(db_url)
    return engines[db_url]


def add_alarm(visitor_id, camera_id):
    """
    添加报警接口

    args:
        visitor_id: visitor 表中的 id
        camera_id: 视频流的来源 id
    """
    eng = get_engine()
    with eng.connect() as conn:
        ins = Alarm.insert().values(visitor_id=visitor_id, camera_id=camera_id)
        conn.execute(ins)


if __name__ == "__main__":
    eng = create_engine(db_url)
    table = Table('visitors', MetaData(bind=eng))
    import ipdb
    ipdb.set_trace()
