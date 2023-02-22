# Databricks notebook source
# MAGIC %md
# MAGIC ![FFmpeg](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/FFmpeg_Logo_new.svg/1024px-FFmpeg_Logo_new.svg.png)
# MAGIC 
# MAGIC # Get Started with FFmpeg for audio manipulation
# MAGIC 
# MAGIC - Detailed documentation on
# MAGIC   - [FFmpeg docs](https://ffmpeg.org/documentation.html)
# MAGIC   - [Audio Channel Manipulation docs](https://trac.ffmpeg.org/wiki/AudioChannelManipulation) 
# MAGIC   - [Python bindings for FFmpeg](https://kkroening.github.io/ffmpeg-python/) 

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

# COMMAND ----------

# DBTITLE 1,Install FFmpeg Python wrapper
# MAGIC %pip install ffmpeg-python

# COMMAND ----------

# DBTITLE 1,Load Libraries
# import library for manipulating audio content
import ffmpeg

# COMMAND ----------

# DBTITLE 1,Print Media Stats
# MAGIC %sh
# MAGIC 
# MAGIC ffmpeg -i /dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav -af astats -f null -

# COMMAND ----------

# DBTITLE 1,Split Left/Right Channels from Stereo Audio
# MAGIC %sh
# MAGIC 
# MAGIC # It should be possible to combine the next two commands into one, but for areason I ignore, the execution kept halting and I had to clear the state :/
# MAGIC 
# MAGIC # for a reason, ffmpeg can't write into dbfs, TODO: confirm this is related to padding and seeking
# MAGIC # we write to local fs first and then move to dbfs / cloud storage
# MAGIC 
# MAGIC # Isolate left channel
# MAGIC ffmpeg -i /dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav -map_channel 0.0.0 /tmp/bw/0638_left.wav
# MAGIC 
# MAGIC # Isolate right channel
# MAGIC ffmpeg -i /dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav -map_channel 0.0.1 /tmp/bw/0638_right.wav

# COMMAND ----------

# DBTITLE 1,Split Left/Right Channels Programmatically
input = '/dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav'

ch_right = ffmpeg.input(input)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FR'
        )

ffmpeg.output(ch_right, '/tmp/bw/0638_prog_right.wav').run(capture_stdout=True, capture_stderr=True)

ch_left = ffmpeg.input(input)['a:0'].filter(
            'channelsplit',
            channel_layout='stereo',
            channels='FL'
        )

ffmpeg.output(ch_left, '/tmp/bw/0638_prog_left.wav').run(capture_stdout=True, capture_stderr=True)

# COMMAND ----------

# DBTITLE 1,Trim Audio File => Trick to generate many that are almost the same but different :)
file = "/dbfs/FileStore/Users/bachar.wehbi@databricks.com/talkbank/0638.wav"

ch = ffmpeg.input(input)

ch_trimmed = ch.filter('atrim', start=10) # trim the first 10 seconds

# for a reason, ffmpeg can't write into dbfs, TODO: confirm this is related to padding
# we write to local fs first and then move to dbfs / cloud storage
ffmpeg.output(ch_trimmed, f'/tmp/bw/0638_prog_trimmed.wav').run(capture_stdout=True, capture_stderr=True)

# COMMAND ----------

# DBTITLE 1,Verify files were created :)
# MAGIC %fs ls file:/tmp/bw

# COMMAND ----------

# DBTITLE 1,Copy files from tmp to dbfs
# MAGIC %fs cp -r file:/tmp/bw/ dbfs:/FileStore/Users/bachar.wehbi@databricks.com/my_awesome_folder/
