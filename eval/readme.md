# Running the Script

To run the evaluation script, follow these steps:

1. Navigate to the directory containing the eval.py script.
2. Use the command line to run the script with the necessary arguments. Below is an example command:

```bash
python3 eval.py --write_predictions
```

3. Adjust the paths and parameters according to your setup:
- --coor2Ddir and --coor3Ddir for coordinate data.
- --layout2Ddir and --layout3Ddir for layout images.
- --threshold to set the object tracking accuracy threshold.
- --accelerator to choose between CPU and edgeTPU processing.


# Leaderboard
Below is the leaderboard for the metric scores obtained from different model evaluations. The initial baseline is provided by the dot-vision-v1 model. Future contributions that improve upon these scores will be added to the top of this table.

| Method       |  Model            |  Accelerator | Accelerator Detail | MSE         | Average FPS | STDev FPS   |
| -----------  |  -----------      |  ----------- | -----------        | ----------- | ----------- | ----------- |
| dot-vision-v1|  SSD MobileNet V1 |  cpu         | Apple M1               | 0.32        | 10.8        | 4.1         |

The best scores in the leaderboard will be merged into main branch.

# Note
When running the script, ensure that all paths and dependencies are correctly set up according to your local environment. The script is designed to be modular, and different parameters can be tuned as per the requirements of the evaluation.
