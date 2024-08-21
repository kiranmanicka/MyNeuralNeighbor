 ### This is my implementation of Neural Neighbor Style Transfer(https://arxiv.org/pdf/2203.13215) as well as Vanilla Neural Style Transfer.

I have included Jupyter Notebooks of NST and NNST as well as python scripts:

To interface with nst.py:

python nst.py -c=[name of content image in content directory] -s=[name of style image in style directory]

or

python nst.py --content [name of content image in content directory] --style [name of style image in style directory]

When using nst.py or nnst.py, it will store the resulting blended image into either nst_blended or nnst_blended.
I have already included some example blended images.

### Neural Style Transfer

Neural Style Transfer is a popular method to transfer the style of one image to a content image. This is done by constructing a loss function that takes into account a content loss and a style loss. The content loss is just the Mean Square Error between the activations of the content image and the image being generated at a particular layer. The style loss is the Mean Square Error between the Gram Matrices of the style image and the image being generated at various layers. This technique works well, however it really lacks the nuance that is expected. For example, if the style image is a pencil drawing of a child balloons, the resulting image will seem to have a bunch of black and white circular objects in it. The algorithm doesn't learn how to create the generated image in the drawing style of style image, but instead seems to try its best overlaying the two images. I believe this is because of the nature of the loss function which contains two conflicting elements which acts like a seesaw.

### Neural Neighbor Style Transfer

Neural Neighbor Style Transfer builds on NST in a significant way, instead of constructing a loss function that behaves like a seesaw, what it does is it grabs the activations of the style image at various layer, bilinearly interpolates and concatentates them. This is what the authors refer to as hypercolumns. You then do this same process for the content image. Now you perform KNN with each content hypercolumn amongst the set of style hypercolumns, replacing the content hypercolumns with its appropriate nearest style hypercolumn. This now becomes your target, and as you are generating your output you are attempting to get your activation as close to this target as possible. This seems to be a better approach than Vanilla NST because it gives the algorithm one target instead of two and because intuitively you are trying to match activations that not only occur in the style image but are also similar to the activation in the content image. In theory this should lead to higher fidelity.

### Observations:
All of the generated images are stored in nnst_blended and nst_blended. From what I can see, vanilla NST definitely does a better job at capturing high level features like larger objects while NNST does a better job at capturing the textures. There are some examples where NNST definitely outperform NST because the given style image does not really contain any large object but more of a recurring pattern. 

### Limitations:
There are many examples within the NNST blended images that don't seem to be great. They do capture the textures well but they fail to capture the essence of the style image. In many of the images the texture comes through however the color looks faded. I believe this is because color is a low level feature and low level features are a small fraction of the hypercolumn compared to the high level features meaning that gradient descent will prefer to optimize over the higher level features. I believe this occurred because I did not implement the feature splitting preocess that the authors recommended. The authors in the paper recommend to match features from the content image to the style image per layer instead of concatentating everything and then trying to match. I believe this helps because it allows the algorithm more flexibility to create what it needs to. Just because a certain content hypercolumn matches well with a style hypercolumn doesn't necessarily mean all of the subfeatures match well. Feature splitting allows each subfeature to match well ensuring that the content image is transformed to match the low level as well as the high level features of the style image. Another thing the authors implemented that I did not was recomputing the target after every iteration. This helps because features are matched relative the current output and not what the output started off as (the content image). This means the algorithm is a receiving a strong and updated signal every iteration, instead of once at the beginning which could have maybed helped with the color scheme. I could not implement these aspects of the authors paper due to lack of compute. 
