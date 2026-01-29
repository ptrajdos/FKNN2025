ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))
DATAFILE=${ROOTDIR}/tsnre_windowed.tar.xz
DATAFILEID=15D-PUeOIHQXyJMhaoDGOxmSfAayMyBgb
DATADIR=${ROOTDIR}/data
VENV_SUBDIR=${ROOTDIR}/venv
EXPERIMENT_SUBDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments
INSTALLATION_LOG=${ROOTDIR}/install.log

SRCDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework
TESTDIR?=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
REQ_FILE=${ROOTDIR}/requirements_dev.txt
INSTALL_LOG_FILE=${ROOTDIR}/install.log
LOGDIR=${ROOTDIR}/testlogs
COVERAGE = coverage

PYTEST=pytest
PYTHON=python
SYSPYTHON=python
PIP=pip
CURL=curl
TAR=tar

VENV_OPTIONS=



.PHONY: all clean

.NOTPARALLEL: run_experiments

create_env: venv data

clean:
	rm -rf ${VENV_SUBDIR}


experiment1: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/channel_knn_weights_ex_1.py

experiment2: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/channel_knn_weights_ex_2.py

experiment3: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/channel_knn_weights_ex_3.py

run_experiments: experiment1 experiment2 experiment3
	@echo "All experiments completed successfully."


venv:

	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install -e . -r ${REQ_FILE} --log ${INSTALLATION_LOG}

profile: venv
	
	. ${VENV_SUBDIR}/bin/activate; ${PYTEST} -n auto --cov-report=html --cov=${SRCDIR} --profile ${TESTDIR}

test:venv
	mkdir -p ${LOGDIR}
	. ${VENV_SUBDIR}/bin/activate; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	. ${VENV_SUBDIR}/bin/activate; ${COVERAGE} html --show-contexts

data:
	mkdir -p ${DATADIR}
	${CURL} -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"
	${TAR} -xvf ${DATAFILE} --directory ${DATADIR}
	
