This repository contains the code to reproduce the results shown in the paper "Set-wise Coordinate Descent for Dual
Asynchronous Decentralized Optimization", submitted for review to IEEE Transactions on Automatic Control.

Below are the instructions to run the code using Google Colab. This requires having a Google account. 

To run the code on Google Colab: 

1. Open Colab and mount your Google Drive folder by running:
   `from google.colab import drive`
   `drive.mount('/content/drive')`
   in an arbitrary cell.
2. A window will pop up asking if you grant the notebook access to your Google Drive files. Click on Connect to Google Drive. A new window will open for you to select the Google account to use. Select your preferred account and click on Authorize. 
3. Copy this GitHub repository to your Google Drive by executing:     
   `%cd /content/drive/MyDrive/Colab\ Notebooks`     
   `!git clone https://github.com/m-costantini/Set-wise_Coordinate_Descent`
   in a new cell
4. You have now a Google Drive folder inside the Colab Notebooks folder named Set-wise_Coordinate_Descent. Open Google Drive and navigate to that folder. Open any of the main notebooks by double-clicking. A new Colab window will open with the notebook ready to execute.
5. To run the code, go to `Runtime >> Run all` or do `Ctrl` + `F9`. You may need to repeat step 2 at this point to grant the notebook of the experiment access to your Google Drive. 


