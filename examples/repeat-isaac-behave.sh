#!/bin/env sh
# Script for runnign `isaacsim-behave` for a number of repetitions,
# specified as a single numeric argument

COUNT=$1
CONF_FILE=$2
MODEL_FILE=$3

ISAAC_VER='4.2.0'
ISAAC_SIM_DIR="/media/ext4-data/share-admin/nvidia/isaac-sim-${ISAAC_VER}"
ISAAC_PYTHON_SH="${ISAAC_SIM_DIR}/python.sh"
alias isaacsim-python-sh="${ISAAC_PYTHON_SH}"
alias isaacsim-pip="${ISAAC_PYTHON_SH} -m pip"
alias isaacsim-ipython="${ISAAC_PYTHON_SH} ${ISAAC_SIM_DIR}/kit/python/bin/ipython"
alias isaacsim-behave="${ISAAC_PYTHON_SH} ${ISAAC_SIM_DIR}/kit/python/bin/behave"

while [ $COUNT -gt 0 ]
do
    # Print the values
    echo "Repeat counter: $COUNT"
    # isaacsim-behave
    isaacsim-behave -D config_file=$CONF_FILE -D model_file=$MODEL_FILE
    # increment the value
    COUNT=`expr $COUNT - 1`
done
