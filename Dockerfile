# docker doesn't have any X11 windowing system (a.k.a GUI)
# therefore downloading stardard version of opencv will result in failure to build
# we need to use the headless version of opencv

FROM python:3.9-slim
LABEL authors="kaorikizuna"

WORKDIR /app
COPY docker_requirements.txt .
COPY . .

RUN pip install -r docker_requirements.txt

EXPOSE 5000
CMD ["python3", "dot_vision.py"]

# run this file with -P, to use the exposed port.