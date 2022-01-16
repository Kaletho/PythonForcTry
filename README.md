# PythonForcTry
My try at calculating and plotting FORCS with python!

Some time ago I wrote this program to read, calculate and plot first order reversal curves (FORCs) from a vibrating sample magnetometer (VSM) at the lab.

Here, you will find an example FORC file of an array of nickel nanowires inside an alumina template (which is kind of a standard) and the python script used to plot the FORCs.
The script was originally written in python 2.7 so some print parenthesis look out of place and I have some comments have been added to the changes to make it work in 3+

There are still things to do, like improving how the "derivatives" are handled at the edges (This is the reason of the red stripe on the top left corner of the plots)

Hope this helps anyone who is trying to understand and implement a FORC script in python
