from typing import Union, Annotated

from fastapi import FastAPI, Form

from pydantic import BaseModel
from modelscope.pipelines import pipeline

import requests
import os
from urllib.parse import urlparse

inference = pipeline('talking-head', model='wwd123/sadtalker', model_revision='v1.0.0') # 请使用最新版的model_revision

app = FastAPI()


@app.get("/")
def read_root():
  return {"Hello": "World"}


class Data(BaseModel):
  source_image_url: str
  driven_audio_url: str

@app.post("/gen/")
def get_gen_video(data: Data):
  source_image_url = data.source_image_url
  driven_audio_url = data.driven_audio_url

  source_image = '/image/' + os.path.basename(urlparse(source_image_url).path) # 请修改成你的实际路径
  driven_audio = '/audio/' + os.path.basename(urlparse(driven_audio_url).path) # 请修改成你的实际路径

  print(source_image)
  print(driven_audio)

  def makeDir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

  makeDir(source_image)
  makeDir(driven_audio)

  response = requests.get(source_image_url)
  with open(source_image, 'wb') as file:
    file.write(response.content)

  response = requests.get(driven_audio_url)
  with open(driven_audio, 'wb') as file:
    file.write(response.content)

  # 其他可选参数
  out_dir = './results/' # 输出文件夹
  kwargs = {
      'preprocess' : 'full', # 'crop', 'resize', 'full'
      'still_mode' : True,
      'use_enhancer' : False,
      'batch_size' : 1,
      'size' : 256, # 256, 512
      'pose_style' : 0,
      'exp_scale' : 1,
      'result_dir': out_dir
  }

  video_path = inference(source_image, driven_audio=driven_audio, **kwargs)
  print(f"==>> video_path: {video_path}")

  # files = {'file': open(video_path, 'rb')}
  #
  # r = requests.post('http://47.122.46.24:8003/api/file/upload/', files=files).json()
  # print(f"url: {r['data']}")

  ext = video_path.rsplit('.', 1)[1]
  updated_file = video_path.rsplit('.', 1)[0] + "_updated" + '.' + ext

  command = 'ffmpeg -i "%s" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k "%s"' % (video_path, updated_file)
  os.system(command)

  files = {'file': open(updated_file, 'rb')}

  r = requests.post('http://47.122.46.24:8003/api/file/upload/', files=files).json()
  print(f"url: {r['data']}")
  video_url = r['data']

  return video_url