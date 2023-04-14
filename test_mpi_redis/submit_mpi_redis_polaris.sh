#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug 
#PBS -A datascience
#PBS -l filesystems=grand:home

cd ${PBS_O_WORKDIR}

set -xe

echo `pwd`
source ../../scalable-bo/build/activate-dhenv.sh

redis-server $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf
mpiprun -np 4 python3 mpi_dbo_with_redis.py
