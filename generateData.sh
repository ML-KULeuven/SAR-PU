env=$1

datasets="20ng Adult BreastCancer Covtype Diabetes ImageSegmentation Mushroom Splice"
for data in $datasets; do
	echo Download and preprocess $data
	jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=$env --execute --clear-output notebooks/data_preprocessing/$data.ipynb 
done

echo Make extended datasets
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=$env --execute --clear-output notebooks/data_preprocessing/Extended\ Data.ipynb 
