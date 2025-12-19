#rm -rf dot_files/*
rm -rf png_files/*

#python3 main.py

set -e

# Check arguments
#if [ "$#" -ne 2 ]; then
    #echo "Usage: $0 <input_dot_folder> <output_png_folder>"
    #exit 1
#f4

input_dir="/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/dot_files/fp_want2"
output_dir="/home/bhavya/cosmos/life/UIUC/academics/coursework/CS521/Project/CS521_TASO/png_files"

# Verify input directory exists
if [ ! -d "$input_dir" ]; then
    echo "❌ Error: input directory '$input_dir' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate over all .dot files
ls "$input_dir/"
for dotfile in "$input_dir"/*; do
	echo $dotfile
    # Skip if no .dot files exist
	[ -e "$dotfile" ] || { echo "No .dot files found in $input_dir"; exit 0; }

    filename=$(basename "$dotfile" .dot)
	echo "$filename"
    output_file="$output_dir/${filename}.png"
	echo "$output_file"

    echo "🧩 Converting $dotfile → $output_file"
    dot -Tpng "$dotfile" -o "$output_file"
done

#echo "✅ Converted $count .dot file(s) to PNG in '$output_dir'"

#dot -Tpng graph0.dot -o graph0.png


