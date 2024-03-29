#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -q prod
# #PBF -q preemptable
#PBS -A datascience
#PBS -l filesystems=grand:home
##PBS -l filesystems=home

set -xe

cd ${PBS_O_WORKDIR}

source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh

# Configuration to place 1 worker per GPU
export NDEPTH=4
export NRANKS_PER_NODE=16
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH
export REDIS_CONF="/lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/redis.conf"

# Set the seed
for seed in 0 1 2 3 4 5 6 7 8 9
do
	# Run DeepHyper AC
	# Setup Redis Database
	export log_dir="dtlz_mpi_logs-AC_id"
	mkdir -p $log_dir
	pushd $log_dir
	redis-server $REDIS_CONF &
	export DEEPHYPER_DB_HOST=$HOST
	popd
	sleep 5
	# Run the DeepHyper script
	mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
	    --depth=${NDEPTH} \
	    --cpu-bind depth \
	    --envall \
	    python dtlz_mpi_solve_AC_id.py $seed
	# Stop the redis server
	redis-cli shutdown
	
	# Run DeepHyper C
	# Setup Redis Database
	export log_dir="dtlz_mpi_logs-C_id"
	mkdir -p $log_dir
	pushd $log_dir
	redis-server $REDIS_CONF &
	export DEEPHYPER_DB_HOST=$HOST
	popd
	sleep 5
	# Run the DeepHyper script
	mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
	    --depth=${NDEPTH} \
	    --cpu-bind depth \
	    --envall \
	    python dtlz_mpi_solve_C_id.py $seed
	# Stop the redis server
	redis-cli shutdown
	
	# Run DeepHyper L
	# Setup Redis Database
	export log_dir="dtlz_mpi_logs-L_id"
	mkdir -p $log_dir
	pushd $log_dir
	redis-server $REDIS_CONF &
	export DEEPHYPER_DB_HOST=$HOST
	popd
	sleep 5
	# Run the DeepHyper script
	mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
	    --depth=${NDEPTH} \
	    --cpu-bind depth \
	    --envall \
	    python dtlz_mpi_solve_L_id.py $seed
	# Stop the redis server
	redis-cli shutdown
	
	# Run DeepHyper P
	# Setup Redis Database
	export log_dir="dtlz_mpi_logs-P_id"
	mkdir -p $log_dir
	pushd $log_dir
	redis-server $REDIS_CONF &
	export DEEPHYPER_DB_HOST=$HOST
	popd
	sleep 5
	# Run the DeepHyper script
	mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
	    --depth=${NDEPTH} \
	    --cpu-bind depth \
	    --envall \
	    python dtlz_mpi_solve_P_id.py $seed
	# Stop the redis server
	redis-cli shutdown
	
	## Run DeepHyper Q
	## Setup Redis Database
	#export log_dir="dtlz_mpi_logs-Q_id"
	#mkdir -p $log_dir
	#pushd $log_dir
	#redis-server $REDIS_CONF &
	#export DEEPHYPER_DB_HOST=$HOST
	#popd
	#sleep 5
	## Run the DeepHyper script
	#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
	#    --depth=${NDEPTH} \
	#    --cpu-bind depth \
	#    --envall \
	#    python dtlz_mpi_solve_Q_id.py $seed
	## Stop the redis server
	#redis-cli shutdown
done
