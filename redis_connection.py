import redis

rds = redis.Redis(host='127.0.0.1', port=6379)


def insert_known_face(key, values):
    rds.rpush(key, values)


def known_face():
    known_list = rds.lrange('known_people', 0, -1)
    known_face_names = []
    for _ in known_list:
        known_face_names.append(_.decode("utf-8"))
    return known_face_names
