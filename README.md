incrLaSVM
=========

Incremental LaSVM

LASVM is an approximate SVM solver that uses online approximation.  
It reaches accuracies similar to that of a real LibSVM after performing a single sequential pass through the training examples.
It supports only binary classification. LaSVM source code is mix of C and C++. 
It can be freely downloaded from   http://leon.bottou.org/_media/projects/lasvm-source-1.1.tar.gz . 

Here, the incremental extension to the online algorithm is carried out. 
TODO : Further integrating SVMSOANAL will be next agenda. Code is already available at http://image.diku.dk/igel/solasvm/2nd-order-LASVM.tar.gz

Paper is available [http://image.diku.dk/igel/paper/SOSMOISVMOaAL.pdf]

License
-------
The LaSVM license ("GNU Public License") is compatible with many free software licenses. The GNU General Public License (GNU GPL or GPL) is the most widely used free software license, which guarantees end users (individuals, organizations, companies) the freedoms to use, study, share (copy), and modify the software. 

Executables
-----------
It provides three executables, namely,
* la_svm 
* la_incr
* la_test
la_svm is used only for 1st time model building. For incrementally building the model based on the 1st model (built using la_svm), la_incr is used. 
la_test is used for predictions.

Examples:
=========

I. Training from scratch
------------------------

For training the model without increment, example of the command that needs to be issued is:

la_svm -c 16 –g 1 –p 2 –m 2000 \<input_file\> \<model_file\>

For training the model incrementally with persistence, example of the command that needs to be issued is:

la_svm –i 1 -c 16 –g 1 –p 2 –m 2000 \<input_file\> \<model_file\>

The files persisted are :
  1.  Model file(name: \<model_file\>)
  2.  History file(name: \<model_file\>.history)
  3.  lasvm file(name: \<model_file\>.lasvm)
  4.  lacache file(name: \<model_file\>.lacache)

For training the model incrementally without persistence, example of the command that needs to be issued is:

la_svm –i 2 -c 16 –g 1 –p 2 –m 2000 \<input_file\> \<model_file\>


II. Incremental training
---------------------
For building the model incrementally with persistence mode the command that needs to be issued is, 

la_incr –i 1 –m 2000 \<input_file\> \<model_file\>

Here, it assumes that above mentioned persistence files are already present. 

For building the model incrementally without persistence mode the command that needs to be issued is, 

la_incr –i 2 –m 2000 \<input_file\> \<model_file\>


III. Predictions
----------------
la_test is used to predict the new unseen data. 

The command to be issued for the same is:

la_test \<input_file\> \<model_file\> \<output_file\>

Usages:
========
Complete usage details can be displayed by invoking the executables.

