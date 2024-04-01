# initialization script that is meant to run on the host machine (your laptop/pc)

mdt shell
rm -rf dot_vision/
git clone https://github.com/alvinwatner/dot-vision
cd dot-vision || exit
chmod +x start.sh
./start.sh
