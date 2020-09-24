# RotNet Fork

This is a fork of the original RotNet to use for NuronWorld. OpenStreetCam images have been bungled by their parent company such that we cannot retrieve the original image directly from EXIF data so we are using a (very slightly) modified version of RotNet to do this

You should use `correct_rotation.py` script to correct the orientation of your own images. You can run it as follows:

`python correct_rotation.py <path_to_hdf5_model> <path_to_input_image_or_directory> -o <path_to_output>`

You should also specify the following command line arguments:
- `-o, --output` to specify the output image or directory.
- `-b, --batch_size` to specify the batch size used to run the model.
- `-l, --log` to specify if you would like the program to log the output to a .csv file so you can retrain the model

