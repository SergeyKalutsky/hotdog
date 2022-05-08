import requests


res = requests.get('https://github.com/SergeyKalutsky/hotdog')
print(res.content)