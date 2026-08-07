#!/bin/bash -x

JOBID=$1
NCPU=$2

export HOME=`pwd`
if [ -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
  export HOME=/afs/cern.ch/user/${USER:0:1}/$USER  # crucial on lxplus condor but cannot set on cmsconnect
fi
env


WORKDIR=`pwd`

# Get arguments
declare -A ARGS
for key in workflow output samplejson year campaign isSyst ttbar_reweights isArray noHist overwrite voms chunk skipbadfiles outputDir remoteRepo; do
    echo $(jq -r ".$key" $WORKDIR/arguments.json)
    ARGS[$key]=$(jq -r ".$key" $WORKDIR/arguments.json)
done

# Set up mamba environment
## Interactive bash script with fallback pointing to $HOME, hence setting $PWD of worker node as $HOME
export HOME=`pwd`

echo "Setting up mamba environment"
if [[ ${ARGS[remoteRepo]} != "" ]]; then
    echo "remoteRepo is set to ${ARGS[remoteRepo]}"
    for i in {1..10}; do
        wget -L micro.mamba.pm/install.sh
        if [ $? -eq 0 ]; then
            break
        fi
	echo "Failed attempt #"$i" to download mamba installer. Will retry."
        sleep 30
    done
    chmod +x install.sh
    ## FIXME parsing arguments does not work. will use defaults in install.sh instead, see https://github.com/mamba-org/micromamba-releases/blob/main/install.sh 
    ## Tried solutions listed in https://stackoverflow.com/questions/14392525/passing-arguments-to-an-interactive-program-non-interactively
    ./install.sh <<< $'bin\nY\nY\nmicromamba\n' 
    source .bashrc
fi

export PATH=$WORKDIR:$PATH

if [ ! -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
    ## install necessary packages if on cmsconnect
    micromamba install -c conda-forge jq --yes
fi

# Create base env with python=3.10 and setuptools<=70.1.1
micromamba activate 
micromamba install python=3.10 -c conda-forge xrootd --yes
micromamba activate base
micromamba install setuptools=70.1.1

# Install BTVNanoCommissioning
mkdir BTVNanoCommissioning
cd BTVNanoCommissioning
if [ ! -f $WORKDIR/BTVNanoCommissioning.tar.gz ]; then
    ## clone the BTVNanoCommissioning repo only, no submodule
    git clone ${ARGS[remoteRepo]} .
else
    tar xaf $WORKDIR/BTVNanoCommissioning.tar.gz
fi

pip install -e .

## other dependencies
pip install psutil

# Build the sample json given the job id
python -c "import json, os; flname = 'split_samples.json' if os.path.isfile(f'$WORKDIR/split_samples.json') else 'split_samples_resubmit.json';  json.dump(json.load(open(f'$WORKDIR/{flname}'))['$JOBID'], open('$WORKDIR/sample.json', 'w'), indent=4)"
cp $WORKDIR/sample.json $WORKDIR/BTVNanoCommissioning/sample.json

ls -lah $WORKDIR
ls -lah $WORKDIR/BTVNanoCommissioning

# Unparse arguments and send to runner.py
OPTS="--wf ${ARGS[workflow]} --year ${ARGS[year]} --campaign ${ARGS[campaign]} --chunk ${ARGS[chunk]}"
if [ "${ARGS[voms]}" != "null" ]; then
    OPTS="$OPTS --voms ${ARGS[voms]}"
fi
if [ "${ARGS[isSyst]}" != "false" ]; then
    OPTS="$OPTS --isSyst ${ARGS[isSyst]}"
fi
if [ "${ARGS[ttbar_reweights]}" != "none" ]; then
    OPTS="$OPTS --ttbar-reweights ${ARGS[ttbar_reweights]}"
fi
for key in  isArray noHist overwrite skipbadfiles; do
    if [ "${ARGS[$key]}" == true ]; then
        OPTS="$OPTS --$key"
    fi
done
OPTS="$OPTS --output ${ARGS[output]//.coffea/_$JOBID.coffea}"  # add a suffix to output file name
OPTS="$OPTS --json sample.json"  # use the sample json for this JOBID

# Check the number of CPUs requested and set the worker accordingly.
# If nCPU > 1, use futures executor with nCPU workers. If nCPU = 1, use iterative executor with 1 worker.
if [ $NCPU -gt 1 ]; then
    OPTS="$OPTS --worker $NCPU"  # use number of worker = nCPU
    OPTS="$OPTS --executor futures"
else
    OPTS="$OPTS --worker 1"  # use number of worker = 1
    OPTS="$OPTS --executor iterative"
fi

# Launch
echo "Now launching: python runner.py $OPTS"
python runner.py $OPTS
RUNNER_RC=$?
if [ $RUNNER_RC -ne 0 ]; then
    echo "ERROR: runner.py exited with code $RUNNER_RC, skipping output transfer" >&2
    exit $RUNNER_RC
fi

# Transfer output
## Reconstruct the directory names runner.py used rather than globbing them.
## `hists_*` only matched the default `hists.coffea` output name, and `arrays_*`
## happily matched the empty directory runner.py pre-creates for --isArray, so a
## workflow that writes its root files elsewhere looked like a successful job.
OUTNAME=${ARGS[output]//.coffea/_$JOBID.coffea}
HISTDIR=${OUTNAME%%.*}    # hists_<JOBID>
ARRAYDIR=arrays_$HISTDIR  # arrays_hists_<JOBID>

transfer() {
    local src=$1
    if [ ! -d "$src" ]; then
        echo "ERROR: expected output directory $src does not exist" >&2
        return 1
    fi
    if [[ ${ARGS[outputDir]} == root://* ]]; then
        xrdcp --silent -p -f -r "$src" ${ARGS[outputDir]}/
    else
        mkdir -p ${ARGS[outputDir]}
        ## no -p: preserving ownership/timestamps fails on EOS-backed mounts
        cp -f -r "$src" ${ARGS[outputDir]}/
    fi
}

if [ "${ARGS[noHist]}" != true ]; then
    transfer "$HISTDIR" || exit 1
fi

if [ "${ARGS[isArray]}" == true ]; then
    ## runner.py always creates $ARRAYDIR up front, so an empty one means the
    ## workflow wrote its root files somewhere the transfer cannot see. Those
    ## files die with the condor sandbox, so fail instead of reporting success.
    ## Known offenders: BTA_producer and BTA_ttbar_producer write to
    ## ./<dataset>[_<shift>]/, sf_ttdilep_kin writes to its own out_dir_base.
    if [ -z "$(ls -A "$ARRAYDIR" 2>/dev/null)" ]; then
        echo "ERROR: $ARRAYDIR is empty. This workflow does not write arrays through" >&2
        echo "       utils/array_writer.py, so its root files cannot be transferred." >&2
        ls -lah . >&2
        exit 1
    fi
    transfer "$ARRAYDIR" || exit 1
fi

### one can also consider origanizing the root files in the subdirectory structure ###
# for filename in `\ls *.root`; do
#     SAMPLENAME=$(echo "$filename" | sed -E 's/(.*)_f[0-9-]+_[0-9]+\.root/\1/')
#     # SAMPLENAME=$(echo "$filename" | sed -E 's/(.*)_[0-9a-z]{9}-[0-9a-z]{4}-.*\.root/\1/')
#     xrdcp --silent -p -f $filename ${ARGS[outputDir]}/$SAMPLENAME/
# done

touch $WORKDIR/.success
