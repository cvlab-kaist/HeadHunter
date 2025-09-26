for file in *.pdf; do
    # pdf to png
    magick -density 600 "$file" -quality 100 "${file%.pdf}.png"
    # png to webp
    cwebp -q 100 "${file%.pdf}.png" -o "${file%.pdf}.webp"
done