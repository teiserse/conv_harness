readarray -t tests < tests
for impl in impls/*.c
do
	currname="${impl#impls/}"
        currname="${currname%.c}"

gcc -O3 -msse4 -pthread $impl

index=0
while ! [[ ${tests[index]} = "" ]]
do
	testname="${tests[index]}"
	((index = index + 1))
	config="${tests[index]}"
	((index = index + 1))
	length="${tests[index]}"
	((index = index + 1))

	RESULT_FILE="results/${1}-${currname}-${testname}.txt"

if [[ -e $RESULT_FILE ]] ; then
        echo "'$RESULT_FILE' already exists, not running associated test."
else

echo "Running on: $(hostname). Values are in microseconds." >> $RESULT_FILE
echo "Test running using parameters \"$config\" for $length iterations:" >> $RESULT_FILE

run=1
while [[ $run -le $length ]]
do
        echo "$currname - $testname: Running iteration $run out of $length."
        res=`./a.out $config | grep Team`
        if ! [[ $res =~ ^.*\ ([0-9]+)\ .*$ ]] ; then
                echo $res
        else
                echo ${BASH_REMATCH[1]} >> $RESULT_FILE
        fi
        ((run = run + 1))
done
fi

done

done
