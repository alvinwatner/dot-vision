# initialization script that is meant to run on the coral board

rm -rf dot_vision/
git clone https://github.com/alvinwatner/dot-vision
cd dot-vision || exit
chmod +x start.sh
./start.sh
