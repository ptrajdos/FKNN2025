ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))
DATAFILE=${ROOTDIR}/MK_10_03_2022.tar.xz
DATAFILEID=1bvoruWBz7YszlbrIyUs_ccxo80lbyecf
DATADIR=${ROOTDIR}/data
VENV_SUBDIR=${ROOTDIR}/venv
EXPERIMENT_SUBDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments
UPGRADE_LOG=${ROOTDIR}/upgrade.log
INSTALLATION_LOG=${ROOTDIR}/install.log

PYTHON=python
PIP=pip
CURL=curl
TAR=tar



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

	${PYTHON} -m venv ${VENV_SUBDIR}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install  pip setuptools wheel --upgrade --log ${UPGRADE_LOG}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install -e . --log ${INSTALLATION_LOG}

data:
	mkdir -p ${DATADIR}
	${CURL} -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"
	${TAR} -xvf ${DATAFILE} --directory ${DATADIR}
	
