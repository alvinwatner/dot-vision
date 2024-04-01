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
  python3 TFLite_DirectionTracker.py
```
