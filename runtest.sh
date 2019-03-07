config="1920 1080 3 3 5"
length=250

if [[ $# = 1 ]] ; then
	RESULT_FILE=$1
else
	RESULT_FILE=results.txt
fi

if [[ -e $RESULT_FILE ]] ; then
	echo "File '$RESULT_FILE' already exists, not running tests."
	exit 1
fi

echo "Running on: $(hostname). Values are in microseconds." >> $RESULT_FILE
echo "Test running using parameters \"$config\" for $length iterations:" >> $RESULT_FILE

run=1
while [[ $run -le $length ]]
do
	echo "Running iteration $run out of $length."
	res=`./a.out $config | grep Team`
	if ! [[ $res =~ ^.*\ ([0-9]+)\ .*$ ]] ; then
		echo $res
	else
		echo ${BASH_REMATCH[1]} >> $RESULT_FILE
	fi
	((run = run + 1))
done
