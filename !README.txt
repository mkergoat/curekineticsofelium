+++++--------------------------------+++++
 +++----------------------------------+++
  + How to use the set of Python codes +
 +++----------------------------------+++
+++++--------------------------------+++++
        Mathieu Kergoat, 08/22/2019
    mathieu.kergoat@ensta-bretagne.org

0/ You do not have Python ?
---------------------------
Re-write the code in Matlab :-)
You shouldn't need to edit the code (or only for minor adjustements).
Python is pretty straightforward and the basics are easy to learn. Check out the NUMPY, SCIPY and MATPLOTLIB libraries.
To get a Python distribution and an editor, go to https://www.python.org/downloads/ and download Python 3.6.
Once installed, launch IDLE3.6, open the code you wish to open with FILE / OPEN and then press F5: the code will launch.
Then instructions will be shown in another window (Shell).
NB: another method exist. You can launch the code in a cmd window by simply typing filename.py in the right directory.

1/ Formatting the data
----------------------
The DSC software gives a file that can be transformed to an Excel file. Once this is done, convert the Excel to a TEXT DOS file. 
NB: the Python code could be re-written to directly take the data for the raw file.

2/ Using bd.py
--------------
bd.py fits a Borchardt and Daniel (n-th order) model to a set of given data according to ASTM E2041.
To use it you can either:
- prepare a set-up file. It must be written in this pattern (see file su.txt)
			~~~~~~~~~~~~~~~~~~
		***SETUP FILE***
		File-name.extension
		Path-to-directory
		n (Answer to "Want to see the sample information ? y = yes, n= no")
		n (Answer to "Want to see the temperature program ? y = yes, n= no")
		27.5 (Start temperature of the exotherm in Degrees C)
		72.5 (Stop temperature of the exotherm in Degrees C)
			~~~~~~~~~~~~~~~~~~
 The code will ask for the name of the file when launched. Don't forget the extension !! 
- answer to the questions asked for the code. The questions are in fact those you would have answered in the set-up file

Hitting F5 in the Idle window will launch the code. The rest is pretty much straightforward.

bd.py will give you the Borchardt and Daniel model parameters. 
Then, use integration-BD to simulate the cure

3/ Using bd_integration.py 
--------------------------
Open the file. Fill in the blanks at 
***
N (number of time steps for the Euler resolution of the ODE)
dt (time step ---------------------------------------------)
A (pre-exponential factor)
Ea (Activation energy)
n (model order)
***
You can modify values of Temperatures at which the cure is simulated by changing values in array T
Press CTRL+S then F5 and the plots should appear.

4/ Using ks.py
--------------
bd.py fits an autocatalytic, diffusion-enhanced model to a set of data.
As for bd.py you can either use a setup-file (see sui.txt) or answer the questions the code will ask. 
/!\ THE SET UP FILE NEEDS ONE MORE LINE: the value of the total heat of reaction evaluated in Dynamic mode (Delta H) in mJ/mg (or J/g) -- i.e. the NORMALIZED value.
Open code, press F5
The values for the parameters identified are stored in a file named res.txt under the following patern:
			~~~~~~~~~~~~~~~~~~	
		SampleName
		m
		n
		Optimal Parameters
		List containing the constants values
		List containing the Standard Deviations for each value
			~~~~~~~~~~~~~~~~~~
NB: res.txt is stored in the directory indicated in the set-up file/when answering code questions.
Now you should switch to ks_arr.py

5/ Using ks_arr.py
------------------
/!\ PRIOR TO USING ks_arr.py: Create a file named 'resultats.txt' (results in French ;-)) in the same directory as the code. In this file you should write down constants in the following order:
			~~~~~~~~~~~~~~~~~~	
		RUN K1 K2 ALPHAF B ISOT DUMMY
		RunNumber K1Value K2Value alphafValue bValue IsothermalRunTemperature 0
		.
		.
		.
			~~~~~~~~~~~~~~~~~~	
Now open ks_arr and press F5. Values of the constants and the stdev should appear in the shell. Write them down and open ks_integration.py

5/ Using ks_integration.py
------------------
Open the file. Fill in the blanks at 
***
N (number of time steps for the Euler resolution of the ODE)
dt (time step ---------------------------------------------)
A1 (pre-exponential factor for K1)
Ea1 (Activation energy for K1)
A2 (pre-exponential factor for K1)
Ea2 (Activation energy for K1)
m (partial order)
n (partial order)
***
You can modify values of Temperatures at which the cure is simulated by changing values in array T
Press CTRL+S then F5 and the plots should appear.

There you are :-)
