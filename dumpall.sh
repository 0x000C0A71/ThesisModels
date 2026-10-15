

# Train all models
python Testbench.py --random || exit 1

DUMP_DIR="$(pwd)/dump"
HLO_DIR="$(pwd)/hlo"

mkdir -p "${DUMP_DIR}"
mkdir -p "${HLO_DIR}"

python Testbench.py --models | while read model; do
	rm -f $DUMP_DIR/*  || exit 2
	CUDA_VISIBLE_DEVICES="" XLA_FLAGS="--xla_dump_to=${DUMP_DIR}" python Testbench.py $model || exit 3
	candidates=( $(ls $DUMP_DIR/*before_optimizations.txt) )
	if [[ 1 -eq ${#candidates[@]} ]]; then
		cp "${candidates[0]}" "${HLO_DIR}/${model}.hlo" || exit 4
	else
		echo Expected 1 candidate for model $model but got
		echo $candidates
		exit 5
	fi
	rm -f $DUMP_DIR/* || exit 6
done


