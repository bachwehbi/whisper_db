# Databricks notebook source
# MAGIC %md
# MAGIC ![openAI](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/1280px-OpenAI_Logo.svg.png)
# MAGIC # Get Started with Whisper a transcribing model
# MAGIC 
# MAGIC - Detailed documentation on [Open AI whisper](https://openai.com/blog/whisper/)
# MAGIC - Based on [MlFlow pyfunc wrapper](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) 

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

pip install git+https://github.com/openai/whisper.git

# COMMAND ----------

# MAGIC %run ./whisper_mmfpeg_utils

# COMMAND ----------

# DBTITLE 1,Generating the drop down for the model above
dbutils.widgets.dropdown("model_size", "base", ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'], label='Whisper Model Size')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Working with `whisper`

# COMMAND ----------

model_size = dbutils.widgets.get('model_size')
model_whisper = build_whisper_model(model_size)

# COMMAND ----------

experiment_path = get_experiment_path(f"whisper model {model_size}")

print(experiment_path)

# COMMAND ----------

# DBTITLE 1,Register the Model as MLFlow Experiment
import mlflow, logging

mlflow.autolog(log_models=False)

try:
  experiment_setup = mlflow.create_experiment(experiment_path)
  mlflow.set_experiment(experiment_id = experiment_setup.experiment_id)
except:
  print('run experiment already created...\n searching experiment and making it active...')
  experiment_definition = mlflow.get_experiment_by_name(experiment_path)
  mlflow.set_experiment(experiment_id = experiment_definition.experiment_id)

# Create the experiment run to save the model in
run = mlflow.start_run(run_name=f'whisper_{model_size}')

reqs = mlflow.pytorch.get_default_pip_requirements() + ["git+https://github.com/openai/whisper.git", "ffmpeg-python"]

try:
  mlflow.pyfunc.log_model(
    artifact_path="whisper_model", 
    python_model=WhisperModelWrapper(model_whisper), 
    pip_requirements=reqs
  )
  mlflow.end_run("FINISHED")
  print("---- model registered with success ---")
except Exception as e:
  mlflow.end_run("FAILED")
  print("!!! model registered with error: ", str(e))


# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/whisper_model"
mv = mlflow.register_model(model_uri, "whisper_model_bw")
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))

# COMMAND ----------


