
#include "backprop.h"
#include <vector> 
#include <iostream> 
#include<iomanip> //输出精度控制

void main(void)
{
  ///////////////////////
  //0.BPNN parameters ///
  ///////////////////////
  //note: all the bnpp data vector starts from 1.
  const int n_in=8;   //adapt to train data
  const int n_hid=3;
  const int n_out=8;  //adapt to train data
  const int train_iterations = 500000;
  const double bpnn_learning_rate = 0.3;
  const double bpnn_momentum = 0.3;
  
  bpnn_initialize();
  BPNN* net = bpnn_create(n_in,n_hid,n_out);
 
  double out_err = 0;
  double hid_err = 0;

  ////////////////////////////////
  //1.training data generation ///
  ////////////////////////////////
  std::cout<<"1.Generating Train Data:"<<std::endl;
  int training_size = 8;
  struct train_data{ int input[n_in];int target[n_out];};
  std::vector<train_data> train_sample;
  for(int i=0;i<training_size;i++){
	  train_data temp;
	  for(int j=0;j<n_in;j++){
		temp.input[j] = (j==i)?1:0; 
	  }
	  for(int j=0;j<n_out;j++){
		  temp.target[j] = (j==i)?1:0; 
	  }
	  train_sample.push_back(temp);

	  //print training data
	  {
		  std::cout<<"In\t";
		  for(int j=0;j<n_in;j++){
			  std::cout<<train_sample.at(i).input[j]<<" ";
		  }
		  std::cout<<std::endl;
		  std::cout<<"Out\t";
		  for(int j=0;j<n_out;j++){
			  std::cout<<train_sample.at(i).target[j]<<" ";
		  }
		  std::cout<<std::endl;std::cout<<std::endl;
		  //print training data END
	  }
  }

  ////printf hidden unit
  //std::cout<<"hidden unit"<<std::endl;
  //for(int j=0;j<n_hid;j++)
	 // std::cout<<net->hidden_units[j+1]<<" ";
  //std::cout<<std::endl;


  ///////////////////
  //2. train ////////
  ///////////////////
  // sto
  std::cout<<"2.training "<<std::endl;
  std::cout<<"training iteration:"<<train_iterations<<std::endl;
  for(int epochs = 0; epochs<train_iterations;epochs++){
	  for(int i=0;i<training_size;i++){
	  //put bpnn input
		  for(int j=0;j<n_in;j++){
			 net->input_units[j+1] =  train_sample[i].input[j];
		  }
		  for(int j=0;j<n_out;j++){
			  net->target[j+1] =  train_sample[i].target[j];
		  }
	  //END
	  bpnn_train(net, bpnn_learning_rate, bpnn_momentum, &out_err, &hid_err);
	  }
  }
  std::cout<<"training END"<<std::endl;
  std::cout<<"out error: "<<out_err<<"	hiden error: "<<hid_err<<std::endl<<std::endl;
  
  ////printf hidden unit
  //std::cout<<"hidden unit"<<std::endl;
  //for(int j=0;j<n_hid;j++)
	 // std::cout<<net->hidden_units[j+1]<<" ";
  //std::cout<<std::endl;
  

  ///////////////////
  //3. test /////////
  ///////////////////
  std::cout <<std::setiosflags(std::ios::fixed);  //只有在这项设置后，setprecision才是设置小数的位数。
  std::cout<<"3.test model:"<<std::endl;
  std::cout<<"input units\t";
  for(int j=0;j<n_in;j++){
	  net->input_units[j+1] = train_sample[0].input[j];
	  std::cout<< std::setprecision(2)<<net->input_units[j+1]<<" ";
  }
  std::cout<<std::endl;
  
  bpnn_feedforward(net);

  std::cout<<"hidden units\t";
  for(int i=0;i<net->hidden_n;i++){
	  std::cout<< std::setprecision(2)<<net->hidden_units[i+1]<<" ";
  }
  std::cout<<std::endl;
  std::cout<<"output units\t";
  for(int i=0;i<n_out;i++){
	  std::cout<< std::setprecision(2)<<net->output_units[i+1]<<" ";
  }

  std::cout<<std::endl;
  system("pause");
}