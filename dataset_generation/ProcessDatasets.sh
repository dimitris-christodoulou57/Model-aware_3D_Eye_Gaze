DATA_DIR="/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul"

PATH_SETS="${DATA_DIR}/Datasets"

noDisp=1

mkdir -p "${PATH_SETS}/All"
mkdir -p "${PATH_SETS}/MasterKey"

# Generate pupil and iris fits offline
# Extract images and generate master key
nohup python3 ExtractSequenceWhole.py --noDisp=${noDisp} --path2ds=${PATH_SETS} > OpenEDS_sequence__whole_status.out
