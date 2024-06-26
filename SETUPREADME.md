# Deep learning package installation guide


### 1. Installing Anaconda

> Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS.


a) **Installing Anaconda on windows**
<br>
1. [Anaconda Window Installation guide](https://docs.anaconda.com/anaconda/install/windows/)
2. Add the following anaconda path to Enviroment variables
```
1. C:\Users\xxx\anaconda3
2. C:\Users\xxx\anaconda3\Scripts
3. C:\Users\xxx\anaconda3\condabin
4. C:\Users\xxx\anaconda3\Library\bin
```
<strong> Replace xxx with user name </strong>


### 2. Creating virtual enviroment using Anaconda prompt

a) Open **Anaconda Prompt**
> You can search the **Anaconda Prompt** from search bar. It is similar to command prompt

b) conda create --name **newenv**
> newenv in above command is the name of virtual enviroment we want to work on.

c. When conda asks you to proceed, type y:
> proceed ([y]/n)?



### 3. Activating the virtual enviroment
> conda activate newenv

### 4. Instllating Deep learning Packages in virtual enviroment

> This guide is for installation of deep learning framework with our cuda installation.

a) Activate virtual enviroment as instructed above. <br>
b) install packages in virtual enviroment using **pip** command
```
python.exe -m pip install --upgrade pip
mkdir testPython
cd testPython
conda create --name newenv
conda activate newenv
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install tensorflow
sudo pip install jupyter
sudo jupyter-notebook
doskey /history

conda deactivate

```


### 5. Using jupyter-notebook

1. Activate the virtual enviroment as mentioned above.
2. use the command to open jupyter notebook
>jupyter-notebook <br> **Note**: You must install all the required packages for a deep learning program to run.



### 6. Handling Error

1. Error from missing packages
> if you find some package is missing, then install using **pip** command as mentioned above.
