#!/bin/bash
# - 1
# This script requires installing PostgreSQL on your local machine.
# On macOS, you can install it with Homebrew:
# $ brew install postgresql
# or with Conda:
# $ conda install -c conda-forge postgresql

# - 2
# This script also requires installing the correct `sqlalchemy` depenedency:
# $ pip install psycopg2

set -x


#!!! CONFIGURATION - START

#!!! CONFIGURATION - END

export NRANKS=4

export DEEPHYPER_LOG_DIR="optuna_nsgaii_logs"
mkdir -p $DEEPHYPER_LOG_DIR

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="$DEEPHYPER_LOG_DIR/optunadb"
export OPTUNA_DB_HOST="localhost"
initdb -D "$OPTUNA_DB_DIR"
pg_ctl -D $OPTUNA_DB_DIR -l "$DEEPHYPER_LOG_DIR/db.log" start
createdb hpo
### Setup Postgresql Database - END ###

# Run the Search
mpiexec -np 4 python3 dtlz_mpi_solve_optuna.py 0

# Drop and stop the database
dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$DEEPHYPER_LOG_DIR/db.log" stop
