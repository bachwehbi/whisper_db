# Databricks notebook source
# MAGIC %md
# MAGIC ![openAI](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/1280px-OpenAI_Logo.svg.png)
# MAGIC # Get Started with Whisper a transcribing model
# MAGIC 
# MAGIC - Detailed documentation on [Open AI whisper](https://openai.com/blog/whisper/)
# MAGIC - Based on [MlFlow pyfunc wrapper](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) 

# COMMAND ----------

pip install git+https://github.com/openai/whisper.git

# COMMAND ----------

# MAGIC %run ./whisper_mmfpeg_utils

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Working with `whisper`

# COMMAND ----------

# DBTITLE 1,Testing the model works on sample audio files
df = spark.read.format('binaryFile').option("pathGlobFilter", "48*.wav").load("dbfs:/FileStore/Users/bachar.wehbi@databricks.com/talkbank/")

# COMMAND ----------

print(df.count())

# COMMAND ----------

# DBTITLE 1,Run whisper model using a Pandas UDF function directly
df_transcribed = df.withColumn('transcription', transcribe_lr(F.col('content')))

df_transcribed.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save('dbfs:/FileStore/Users/bachar.wehbi@databricks.com/whisper/results/transcribed')

# COMMAND ----------

# DBTITLE 1,Create pandas UDF function from registered MLFlow model
# model reference from the run experiment
logged_model = 'runs:/dcfa35c51cce48108ccb8993bd2ff638/whisper_model'
  
# Load the model as a Pandas UDF
@pandas_udf("array<string>")
def transcribe_from_model(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    model = mlflow.pyfunc.load_model(logged_model)
    #load the model from the registry
    for s in iterator:
        yield model.predict(s)

#save the function as SQL udf
spark.udf.register("transcribe_from_model", transcribe_from_model)

# COMMAND ----------

# DBTITLE 1,Run whisper model using a MLFlow based UDF function
df_transcribed = df.withColumn('transcription', transcribe_from_model(F.col('content')))

df_transcribed.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save('dbfs:/FileStore/Users/bachar.wehbi@databricks.com/whisper/results/transcribedmodel')

# COMMAND ----------

# DBTITLE 1,Reconstruct Conversation
df_bronze = spark.read.format('delta').load('dbfs:/FileStore/Users/bachar.wehbi@databricks.com/whisper/results/transcribedmodel')\
                 .withColumn('fl', F.element_at('transcription', 1))\
                 .withColumn('fr', F.element_at('transcription', 2))\


display(df_bronze.select('path', 'length', 'fl', 'fr'))

# COMMAND ----------

df_silver = process_transciption_output(df_bronze, 'fl', 'fr')

display(df_silver.select('path', 'who', 'trans.start', 'trans.end', 'trans.text').orderBy(col('path').asc(), col("trans.end").asc(), col("trans.start").asc()))

# COMMAND ----------


