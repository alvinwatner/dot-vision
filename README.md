# Dot Vision

An innovative interactive art installation designed for a coffee shop setting. This project aims to engage patrons by visually representing their movements as dots on a display screen. Beyond its artistic appeal, Dot Vision serves as a dynamic marketing tool for Consult NTA, showcasing the company's expertise in technology solutions.



## Features

- **Interactive Art**: Utilizes advanced object detection and computer vision technologies to track the movements of coffee shop patrons, translating these into captivating visual representations (dots) on a digital canvas.
- **Engagement Tool**: Encourages interaction by allowing patrons to influence the art through their movement, creating a unique, dynamic experience.
- **Educational Aspect**: Includes a QR code linked to information about Dot Vision, offering insights into the technology behind the installation and highlighting Consult NTA’s capabilities.



## Deployment

Deployment of this project onto a coral board consists of these steps

Clone the project

```bash
git clone https://github.com/alvinwatner/dot-vision
cd dot-vision
```

Get into your board
```bash
mdt shell
```

Run the `init.sh`

```bash
chmod +x init.sh
./init.sh
```



## Run Locally

Clone the project

```bash
  git clone https://github.com/alvinwatner/dot-vision
```

Go to the project directory

```bash
  cd dot-vision
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the program

```bash
  python3 dot_vision.py --accelerator cpu
```

## Options Summary
Some options (flags) can be passed into the program in order to change its behavior.

```
optional arguments:
  -h, --help                    show this help message and exit
  --vidsource VIDSOURCE         Video source for tracking
  --layout2Ddir LAYOUT2DDIR     2D layout image
  --layout3Ddir LAYOUT3DDIR     3D layout image
  --coor2Ddir COOR2DDIR         2D coordinates data
  --coor3Ddir COOR3DDIR         3D coordinates data
  --live                        Enable live tracking
  --modeldir MODELDIR           Directory containing the detect.tflite and labelmap.txt
  --threshold THRESHOLD         Set the threshold for object tracking accuracy
  --accelerator {cpu,tpu}       Set the accelerator used in object detection
```

The options can be seen by running `dot_vision.py -h`