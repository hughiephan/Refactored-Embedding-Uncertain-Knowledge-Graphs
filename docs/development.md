# How to start development

## Install the following tools

- For working with Pytorch code, you can install Visual Studio Code by following this guide: https://www.geeksforgeeks.org/how-to-install-visual-studio-code-on-windows

- For downloading the code from Github, you can install SourceTree by following this guide: https://confluence.atlassian.com/get-started-with-sourcetree/install-sourcetree-847359094.html

- For managing the Python, PyTorch, and other libraries environment, you can install Anaconda by following this guide: https://www.datacamp.com/tutorial/installing-anaconda-windows

## Working with SourceTree
Download code from GitHub

![image](https://github.com/hughiephan/UKGE/assets/16631121/6ec05dd5-603f-4f10-afc9-c080dab39076)

Add the Repo you want to download, for example: https://github.com/hughiephan/UKGE , then press `Clone`

![image](https://github.com/hughiephan/UKGE/assets/16631121/7dd84aa0-75b3-4aee-9f99-4c822a1d2d64)

Pull the code to make sure the code we are looking at is always the latest. Because sometimes other people will upload new code to this repo and we want to use their new code too, so make sure you pull everytime you start SourceTree

![image](https://github.com/hughiephan/UKGE/assets/16631121/9bd165a6-c048-4ce8-ba2e-7ccbdeee8f81)

Try changing something in the code, I will add the line `This is a test commit` in `run.py`

![image](https://github.com/hughiephan/UKGE/assets/16631121/5030febf-427e-4c45-a7ef-b209acb29636)

To upload the new code, we press `Commit` and then `Stage all`

![image](https://github.com/hughiephan/UKGE/assets/16631121/169e0f5e-e3e8-4565-963c-9f4be5ffb0d0)

Tick on `Push changes immediately to origin/master` and then press `Commit`

![image](https://github.com/hughiephan/UKGE/assets/16631121/0202dbea-8a75-488f-acff-ebdbc7631fd5)

You will see that your code is now successfully uploaded

![image](https://github.com/hughiephan/UKGE/assets/16631121/312880ce-4ef0-49ac-a28d-c4b8fb995437)

Check again in the website and you will see your code is already pushed and uploaded to the Github repo website

![image](https://github.com/hughiephan/UKGE/assets/16631121/bd9ae84d-3a32-4ce7-92e9-4f399ffe5351)

## Working with Visual Studio Code

Click on `File` and `Open Folder`

![image](https://github.com/hughiephan/UKGE/assets/16631121/53611281-1ac5-4672-91fe-5aa7cfbec58d)

Then point to your code Folder and open it

![image](https://github.com/hughiephan/UKGE/assets/16631121/566ebe30-5268-41eb-9fef-c7237b775094)

## Anaconda 

This tool is useful as it allows us to separate many environments. For example, I will have an environment running Python 2 and Pytorch for paper A, and another environment running Python 3 and Keras for paper B

![image](https://github.com/hughiephan/UKGE/assets/16631121/c4fa9cd2-819b-4cfa-8fa5-8a64852bba8b)

Let's create a new `UKGE` environment, click on `Environments`, add the name `ukge` and choose any Python version you want and then click `Create`

![image](https://github.com/hughiephan/UKGE/assets/16631121/f3b9dd86-d651-4488-b8d2-fc24a2bbc6d8)

Then select `Not installed` to access all the packages we haven't installed yet

![image](https://github.com/hughiephan/UKGE/assets/16631121/c7f886d3-9c9b-41bf-91e9-421f6289d45f)

Then type a library you want to install into the search, for example: tensorflow . So basically, you will need to do this for all the libraries that exist in the `requirements.txt` file

![image](https://github.com/hughiephan/UKGE/assets/16631121/4d97013e-e9ef-4434-8ade-cf2a199b3e66)

