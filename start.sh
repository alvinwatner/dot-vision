mdt push . /home/mendel/
mdt exec "cd /home/mendel/ && pip install -r requirements.txt && python3 TFLite_DirectionTracker.py --edgetpu"
mdt pull output.avi .
