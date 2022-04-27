from redis import StrictRedis

redis = StrictRedis(host='110.40.204.239', port=6379, db=0, password='123456')
redis.set('name', 'GEJI')
print(redis.get('name'))