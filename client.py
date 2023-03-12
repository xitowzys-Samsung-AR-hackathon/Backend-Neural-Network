import requests
from io import BytesIO
import base64
from PIL import Image



BASE = "http://127.0.0.1:5000/"

#add single image to database
image = open("C:/Users/danil/Downloads/flaskClient/imagesClient/targets/msg-1667745596-110.jpg", 'rb').read()
response = requests.post(BASE + "add", files={"image":image})
print(response.json()['msg'])

#get best_matching
image = open("C:/Users/danil/Downloads/flaskClient/imagesClient/targets/msg-1667745596-112.jpg", 'rb').read()
response = requests.post(BASE + "find_matching", files={"image":image})
response = response.json()
best_matching = Image.open(BytesIO(base64.b64decode(response['img'])))
best_matching.save('best_matching' + '.' + best_matching.format)

print(response['msg'],response['score'])
