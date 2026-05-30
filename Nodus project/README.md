# Nodus Project How-to

This directory contain the complete Nodus project used to develop the modal choice framework.

- Create a "calibratedod" database in MariaDB/Mysql and create a user named "nodus" with the password "nodus" and grant all privileges to "nodus" for the "calibratedod" database.
- Launch Nodus and open the "CalibratedOD.nodus" project. This will import and display the digitized networks.
- Import the demand matrices using `importdbf tablename` in the Nodus SQL console. The following matrices are provided:
    - od_nuts2_road, od_nuts2_iww and od_nuts2_rail for the European NUTS 2 model
    - od_nuts3_road, od_nuts3_iww and od_nuts3_rail for the Benelux+ NUTS3 model
    - od_del3_road, od_del3_iww and od_del3_rail for the German NUTS3 model
- Merge the modal matrices for each model using the `MergeOD.sql` script in the Nodus SQL console. Instead, you can also directly import the provided already merged matrices od_nuts2, od_nuts3 and od_del3.
- Perform the uncalibrated assignments scenarios 1, 2 and 3, for modes 1, 2 and 3.
- Run the `CreateBiogemeInput.sql` script in the Nodus SQL console. This creates the input tables for the Python/Biogeme script.
- Run the Python/Biogeme script to estimate the parameters.
- Once done, come back in Nodud and run the calibrated assignment scenarios 11, 12 and 13. These scenarios use the modal choice plugin correponding to the modal choice framework presented in this repository.
