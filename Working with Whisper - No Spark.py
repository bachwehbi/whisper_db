# Databricks notebook source
# MAGIC %md
# MAGIC ![openAI](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/1280px-OpenAI_Logo.svg.png)
# MAGIC # Get Started with Whisper a transcribing model
# MAGIC 
# MAGIC - Detailed documentation on [Open AI whisper](https://openai.com/blog/whisper/)
# MAGIC - Based on [MlFlow pyfunc wrapper](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
# MAGIC 
# MAGIC This Notebook includes code snippets showing how to work with Whisper.
# MAGIC 
# MAGIC This is considered as the baseline as we will not distribute the transcription process.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Required libraries
# MAGIC 
# MAGIC #### ffmpeg
# MAGIC the cluster requires `ffmpeg` to be installed. This can be done through init script with the following code:
# MAGIC 
# MAGIC ```
# MAGIC #!/bin/bash
# MAGIC 
# MAGIC echo "this is installing ffmpeg on this cluster"
# MAGIC apt-get update && apt-get install -y ffmpeg
# MAGIC ```
# MAGIC 
# MAGIC #### whisper
# MAGIC 
# MAGIC the library can be installed through pip install on teh cluster or in the notebook
# MAGIC 
# MAGIC `pip install git+https://github.com/openai/whisper.git`
# MAGIC 
# MAGIC > note: with loading audio from binary content does not work with all format. for example it does not work with m4a see [piping with ffmpeg](https://stackoverflow.com/questions/57958480/ffmpeg-cant-stream-aac-files-from-stdin)

# COMMAND ----------

# DBTITLE 1,PIP install the library
pip install git+https://github.com/openai/whisper.git

# COMMAND ----------

# DBTITLE 1,Generating the drop down for the model above
dbutils.widgets.dropdown("model_size", "base", ['tiny', 'base', 'small', 'medium', 'large-v2'], label='Whisper Model Size')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Working with `whisper`

# COMMAND ----------

# DBTITLE 1,Import `whisper` Model
import datetime
import torch
import whisper

ts = datetime.datetime.now().timestamp()

def build_whisper_model(model_size):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  return whisper.load_model(model_size).to(device)
      

# COMMAND ----------

model_size = dbutils.widgets.get('model_size')
model_whisper = build_whisper_model(model_size)

# COMMAND ----------

# DBTITLE 1,Path to Input Audio Data
input_audio_file = '/dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav'

audio_files_path = dbutils.fs.ls("dbfs:/FileStore/Users/bachar.wehbi@databricks.com/talkbank/")

# COMMAND ----------

# DBTITLE 1,Run Whisper model on a file
import json

result = model_whisper.transcribe(input_audio_file)

print('--- this is the output of transcribe ---\n {}'.format(json.dumps(result, indent=2)))

# COMMAND ----------

# DBTITLE 1,Run Whisper Model on a `bytes` object
import numpy as np

with open(input_audio_file, mode="rb") as afile:
  contents = afile.read()
  audio = torch.from_numpy(np.frombuffer(contents, np.int16).flatten().astype(np.float32) / 32768.0)
  result = model_whisper.transcribe(audio)

  print('--- this is the output of transcribe ---\n {}'.format(json.dumps(result, indent=2)))

# COMMAND ----------

# DBTITLE 1,Running the model on files in a folder
# Running the whisper model requires local file paths. Transform the file paths accordingly
# DBFS is mounted into the clusters. This give the possibility to access dbfs files as if they
# were accessible locally.
file_paths = [file.path.replace('dbfs:', '/dbfs') for file in audio_files_path if file.path.endswith('.wav') and '/48' in file.path]

transcriptions = []

i = 1
for f in file_paths:
  ts = datetime.datetime.now().timestamp()
  print(f'starting processing {i}/{len(file_paths)} - file {f} at {ts}')
  i = i + 1
  transcriptions.append(model_whisper.transcribe(f))


# COMMAND ----------


