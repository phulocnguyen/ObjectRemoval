# bash for downloading weights of SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# bash for processing images
python main.py --mode image --input test.jpg --output img_output.jpg
# bash for processing videos
python  main.py --mode video --input vid_test.mp4 --output vid_output.mp4

