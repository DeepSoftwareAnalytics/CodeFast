conda env create -f CodeFast.yml
conda activate CodeFast
pip install -e transformers
pip install -e mxeval
cd mxeval 
bash language_setup/ubuntu.sh
cd ..