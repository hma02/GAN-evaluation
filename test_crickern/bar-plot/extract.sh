folders=(../$1)
subfolders=($2)
# files=(gan.txt wgan.txt lsgan.txt)

for folder in ${folders[@]}
do
	
	for subfolder in ${subfolders[@]}
	do
		
		echo
		echo 'folder '$folder/$subfolder
		files=$(ls $folder/$subfolder)
		for file in ${files[@]}
		do
			echo $file
			if [ -e $folder/$subfolder/$file ]
			then
				grep -oP '(?<=-)[lsganw]+[^\s]*\s[lji][sw]\s[VT][LE][:]\s[0-9]+\.[0-9]+[e-]*[0-9]*\s+\+\-\s+[0-9]+\.[0-9]+[e-]*[0-9]*' $folder/$subfolder/$file	
			fi
		
		done
	done
done