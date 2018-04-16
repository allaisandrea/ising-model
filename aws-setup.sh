#!/usr/bin/bash
sudo apt-get update
sudo apt-get -y install git cmake g++ libboost-all-dev libeigen3-dev
mkdir ${HOME}/.ssh
git clone https://github.com/allaisandrea/ising-model.git
mkdir ising-model/build
mkdir ising-model/data
mkdir ising-model/data/4D
cd ising-model/build
cmake ../src/
make VERBOSE=1
cd -
