#!/bin/bash
DIR=/data/jielin/msmo/video/;
OUT=/home/jielin/claire/video-summ/keyframes/uniform/;
HOMEDIR=$PWD;
sampling_rate="1";

for percent in "15"; do
	echo $percent
	for category in $DIR*/;do 
		echo $category
		for subcat in $category*/;do
			echo $subcat
			for video in $subcat*".mp4";do
				# echo $video
				cd $HOMEDIR
				name=${video##*/};
				folder_name=${name%.mp4};
				# echo $name
				# echo $folder_name
				trunc=$(dirname "$subcat")
				subfolder=$(basename "$trunc")/$(basename "$subcat")
				# mkdir -p $OUT$subfolder/$folder_name
				echo $OUT$subfolder/$folder_name
				# python uniform.py $video $percent $OUT$subfolder/$folder_name/;
				
				# cd ../Evaluation
				# pwd 
				# echo "printing the stuff"
				# echo $filename
				# echo $sampling_rate
				# echo $percent
				# echo $OUT$folder_name"/" $OUT$folder_name"/final_results_uniform_"$percent".txt"
				# python evaluate.py $filename $sampling_rate $percent $OUT$folder_name"/" $OUT$folder_name"/final_results_uniform_"$percent".txt" uniform;
			done
		done
	done
done