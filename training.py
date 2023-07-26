import os
import requests

# zip_path = "vangogh_images.zip"
# zip_filename = zip_path.split("/")[-1]

# # Upload inputs to cloud storage.
# # You can skip this step if your zip file is already on the internet and accessible over HTTP
# upload_response = requests.post(
#     "https://dreambooth-api-experimental.replicate.com/v1/upload/" + zip_filename,
#     headers={"Authorization": "Token " + os.environ["API_TOKEN"]},
# ).json()

# with open(zip_path, "rb") as f:
#     requests.put(upload_response["upload_url"], data=f)
# zip_url = upload_response["serving_url"]

import replicate

zip_url='https://drive.google.com/file/d/1mY14ygDjiymVmwjbgpIXIK5fQJ7d8Hfe/view?usp=sharing'
# Train the model
lora_url = replicate.run(
    "replicate/lora-training:b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc",
    input={"instance_data": zip_url, "task": "style"}
)