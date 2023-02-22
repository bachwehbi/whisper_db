# Databricks notebook source
# MAGIC %md
# MAGIC ## Get Started with Whisper a transcribing model
# MAGIC 
# MAGIC This Notebook includes utility functions to work with `Whisper` and `FFmpeg`.
# MAGIC 
# MAGIC Running this notebook does absolutely nothing other than defining the routies we will be working with.
# MAGIC 
# MAGIC You need to import this notebook to use the functions and definitions as follows:
# MAGIC 
# MAGIC ```
# MAGIC %run ./whisper_mmfpeg_utils
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Working with `whisper`

# COMMAND ----------

# DBTITLE 1,Import `whisper` Model
import datetime
import torch
import whisper

def build_whisper_model(model_size="base"):
  """
  Loads and returns whisper ASR model instance of given ``model_size``.
  The model will be loaded according to available device (GPU or CPU).
  Calling this function the first time will download the model and cache it locally.
  Next calls will use the doanloaded model.
  
  Parameters
  ----------
  model_size: str, default 'base'
    the whisper model size to be loaded. Can be one of ['tiny', 'base', 'small', 'medium', 'large'].
  Returns
  -------
  model : Whisper
    The Whisper ASR model instance
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  return whisper.load_model(model_size).to(device)


def transcribe_from_files(files, model=None, model_size='base'):
  """
  Returns an array of dicts containing the transcriptions results on given input ``files``.
  
  Paramaters:
  files: list[str]
    List of file paths to transcribe
  model : Whisper
    the whisper model to use, if None, a model of given ``model_size`` will be loaded
  model_size: str, default 'base'
    [optional] the whisper model size to be loaded. Can be one of ['tiny', 'base', 'small', 'medium', 'large'].
    if ``model`` is not None, this param will be ignored
  Returns
  -------
  List[dict]
  """
  retval = []
  for file in files:
    ts = datetime.datetime.now().timestamp()
    print(ts, file)
    if model:
      retval.append([ts, model.transcribe(file)])
    else:
      retval.append([ts, build_whisper_model(model_size).transcribe(file)])
  
  return retval
      

# COMMAND ----------

# DBTITLE 1,Transcribe loading with ffmpeg and calling whisper
import ffmpeg
import numpy as np

SAMPLE_RATE = 16000

def load_audio(contents: [str, bytes], sr: int = SAMPLE_RATE, right=False, left=False):
    """
    Open an audio file or bytes content, filter FR or FL channels, and read as mono waveform, resampling as necessary.
    Check out https://github.com/openai/whisper/blob/main/whisper/audio.py#L22-L49 for info about this code.
    
    Parameters
    ----------
    contents: Union[str, bytes]
        The audio file to open or audio bytes stream content
    sr: int
        The sample rate to resample the audio if necessary
    right: bool
      indicates if only front right channel needs to be considered
    left: bool
      indicates if only front left channel needs to be considered

    Returns
    -------
    A torch.Tensor containing the audio waveform.
    """
    file = contents
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        
        if left:
          chan = ffmpeg.input(file, threads=0)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FL'
          )
        elif right:
          chan = ffmpeg.input(file, threads=0)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FR'
          )
        else:
          chan = ffmpeg.input(file, threads=0)

        out, _ = (
            chan.output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return torch.from_numpy(np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0)

# COMMAND ----------

# DBTITLE 1,Create Pandas UDFs to Transcribe in Batch/Streaming Mode
from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F
from typing import Iterator
import pandas as pd
import json
  
# Load the model as a python UDF
@pandas_udf("string")
def transcribe_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

  model = build_whisper_model('large-v2')

  for s in iterator:
    # avoid something like next line as it will hold the decoded audio in memory
    # this can easily lead to out of memory issues whith multiple/long audios
    #decoded_audio = s.map(lambda x: load_audio(x)) # avoid this line
    outputs = s.map(lambda x: json.dumps(model.transcribe(load_audio(x))))
    yield pd.Series(outputs)

    #save the function as SQL udf

spark.udf.register("transcribe_udf", transcribe_udf)

# Load the model as a python UDF
@pandas_udf("string")
def transcribe_left(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

  model = build_whisper_model('large-v2')

  for s in iterator:
    # avoid something like next line as it will hold the decoded audio in memory
    # this can easily lead to out of memory issues whith multiple/long audios
    #decoded_audio = s.map(lambda x: load_audio(x)) # avoid this line
    outputs = s.map(lambda x: json.dumps(model.transcribe(load_audio(x, left=True))))
    yield pd.Series(outputs)

    #save the function as SQL udf

spark.udf.register("transcribe_left", transcribe_left)

# Load the model as a python UDF
@pandas_udf("string")
def transcribe_right(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

  model = build_whisper_model('large-v2')

  for s in iterator:
    # avoid something like next line as it will hold the decoded audio in memory
    # this can easily lead to out of memory issues whith multiple/long audios
    #decoded_audio = s.map(lambda x: load_audio(x)) # avoid this line
    outputs = s.map(lambda x: json.dumps(model.transcribe(load_audio(x, right=True))))
    yield pd.Series(outputs)

    #save the function as SQL udf

spark.udf.register("transcribe_right", transcribe_right)

# Load the model as a python UDF
@pandas_udf("array<string>")
def transcribe_lr(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

  model = build_whisper_model('large-v2')

  for s in iterator:
    # avoid something like next line as it will hold the decoded audio in memory
    # this can easily lead to out of memory issues whith multiple/long audios
    #decoded_audio = s.map(lambda x: load_audio(x)) # avoid this line
    outputs = s.map(lambda x: [json.dumps(model.transcribe(load_audio(x, left=True))), json.dumps(model.transcribe(load_audio(x, right=True)))])
    yield pd.Series(outputs)

    #save the function as SQL udf

spark.udf.register("transcribe_lr", transcribe_lr)

# COMMAND ----------

import mlflow

class WhisperModelWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self, model, sr=SAMPLE_RATE):   
    # instantiate model in evaluation mode
    self.model = model
    self.sr = sr

  @staticmethod
  def load_audio(contents: [str, bytes], sr: int = SAMPLE_RATE, right=False, left=False):
    """
    Open an audio file or bytes content, filter FR or FL channels, and read as mono waveform, resampling as necessary.
    Check out https://github.com/openai/whisper/blob/main/whisper/audio.py#L22-L49 for info about this code.
    
    Parameters
    ----------
    contents: Union[str, bytes]
        The audio file to open or audio bytes stream content
    sr: int
        The sample rate to resample the audio if necessary
    right: bool
      indicates if only front right channel needs to be considered
    left: bool
      indicates if only front left channel needs to be considered

    Returns
    -------
    A torch.Tensor containing the audio waveform.
    """
    file = contents
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        
        if left:
          chan = ffmpeg.input(file, threads=0)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FL'
          )
        elif right:
          chan = ffmpeg.input(file, threads=0)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FR'
          )
        else:
          chan = ffmpeg.input(file, threads=0)

        out, _ = (
            chan.output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return torch.from_numpy(np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0)
  
  def predict(self, context, input_content):

    outputs = input_content.map(lambda x: [
                                json.dumps(self.model.transcribe(WhisperModelWrapper.load_audio(x, left=True))),
                                json.dumps(self.model.transcribe(WhisperModelWrapper.load_audio(x, right=True)))])
    
    return pd.Series(outputs)


# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import from_json, lit, explode, col


transcribe_out_schema = StructType([
  StructField("text", StringType()),
  StructField("segments", ArrayType(StructType([
    StructField("id", IntegerType()),
    StructField("seek", IntegerType()),
    StructField("start", DoubleType()),
    StructField("end", DoubleType()),
    StructField("text", StringType()),
    StructField("tokens", ArrayType(IntegerType())),
    StructField("temperature", DoubleType()),
    StructField("avg_logprob", DoubleType()),
    StructField("compression_ratio", DoubleType()),
    StructField("no_speech_prob", DoubleType())
  ]))),
  StructField("language", StringType())
])

def process_transciption_output(df, trans_left_col_name, trans_right_col_name):
  
  decoded_df = df.withColumn('left_decoded', from_json(trans_left_col_name, transcribe_out_schema))\
                 .withColumn('right_decoded', from_json(trans_right_col_name, transcribe_out_schema))
  
  decoded_left = decoded_df.select('path', 'left_decoded').withColumn('trans', explode('left_decoded.segments')).withColumn('who', lit('left'))
  decoded_right = decoded_df.select('path', 'right_decoded').withColumn('trans', explode('right_decoded.segments')).withColumn('who', lit('right'))

  decoded_all = decoded_left.unionAll(decoded_right)

  return decoded_all


# COMMAND ----------

def get_experiment_path(experiment_name='whisper model'):
  """
  Returns experiment path in the user's workspace.
  Experiments cannot be recorded in Repos
  """

  notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

  # checking if notebook_path in Repos and if so - record experiment at root of Users folder (experiments cannot be recorded in Repos)
  if notebook_path.split("/")[1] == "Repos":
    # will give something like
    # /Users/bachar.wehbi@databricks.com/
    experiment_path = '/Users/' + '/'.join(notebook_path.split("/")[2:3])
  else:
    # will give something like
    # /Users/bachar.wehbi@databricks.com/foldername
    experiment_path = '/' + '/'.join(notebook_path.split("/")[1:-1])

  return f'{experiment_path}/{experiment_name}'
