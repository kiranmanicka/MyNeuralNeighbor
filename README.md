This is my implementation of Neural Neighbor Style Transfer(https://arxiv.org/pdf/2203.13215) as well as Vanilla Neural Style Transfer.

I have included Jupyter Notebooks of NST and NNST as well as python scripts:

To interface with nst.py:

python nst.py -c=[name of content image in content directory] -s=[name of style image in style directory]

or

python nst.py --content [name of content image in content directory] --style [name of style image in style directory]

When using nst.py or nnst.py, it will store the resulting blended image into either nst_blended or nnst_blended.
I have already included some example blended images.

Observations:

Limitations:

