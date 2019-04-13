import setuptools

with open("readme.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='text_representation',  
     version='0.1',
     scripts=['text_representation'] ,
     author="Nikhil Pattisapu",
     author_email="nikhilpriyatam@gmail.com",
     description="A package to extract embeddings from text",
     long_description=long_description,
     long_description_content_type="text/x-rst",
     url="https://github.com/nikhilpriyatam/text_representation",
     packages=setuptools.find_packages(),

     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],

 )
