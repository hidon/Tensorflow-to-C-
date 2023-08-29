In this sharing, I shared how to tensorflow model export to C++ project windows 10.

- Preparing the machine learning model

you have to create to tensorflow model. I did it Anaconda Spyder. to export tensorflow model need to save it any directory. it will use later.

I use this link ([https://www.tensorflow.org/tutorials/keras/regression](url)) to adapted my industrial machine learning applications. My industrial project is about furnace energy consumptions.

 
Build from source on Windows

[https://www.tensorflow.org/install/source_windows](url) you get references this guide.

I install python 3.8.0 and tensorflow-2.9.0 version. Bazelisk is an easy way to install Bazel and automatically downloads the correct Bazel version for TensorFlow.


bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package after this command, executed below commands.

1.  build --config=opt //tensorflow:install_headers (this create include files in bazel-bin\tensorflow folder. This files will use build project on microsoft visual stdio)

2. build --config=opt //tensorflow:tensorflow_cc.lib (this lib will use microsoft visual stdio C++ project)

3. bazel build --config=opt //tensorflow:tensorflow_cc (this create .dll for execute application.)

4. ..\external\com_google_googletest\googlemock\include

5. ..\external\com_google_googletest\googletest\include

4. and 5. folder copy to 1. folder. it is need for microsoft visual stdio build.


Create Microsoft Visual Stdio C++ Project

1. Property Page add Additional Include Directory,Additional library directory and Additional Dependency.

2. Load a SavedModel in C++. ([https://www.tensorflow.org/guide/saved_model](url))
this command list signature key results. you selected it which one of you.
execute
saved_model_cli show --dir C:\Users\...\my_model --tag_set serve
results:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"

I selected "serving_defeault" and Used it in CheckSavedModelBundle function.
Second Command
saved_model_cli show --dir C:\Users\...\my_model --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['normm_input'] tensor_info: (I use "normm_input" in CheckSavedModelBundle function)
      dtype: DT_FLOAT
      shape: (-1, -1)
      name: serving_default_normm_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['Out'] tensor_info: (I use "Out" in CheckSavedModelBundle function)
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict


3. I used "tensorflow/cc/saved_model" project. Below code created from saved_model_bundle_test.cc files. 

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor_testutil.h"

using namespace std;
using namespace tensorflow;


void CheckSavedModelBundle(const string& export_dir, const SavedModelBundle& bundle);

int main()
{
	SavedModelBundle bundle;
	SessionOptions session_options;
	RunOptions run_options;

	const string export_dir = "C:/Users/.../my_model";
	for (int i = 0; i < 100; ++i) {}
	Status st = LoadSavedModel(session_options, run_options, export_dir,{ kSavedModelTagServe }, &bundle);
	if (st.ok())
	{
		CheckSavedModelBundle(export_dir, bundle);
	}

	return 0;
}


void CheckSavedModelBundle(const string& export_dir, const SavedModelBundle& bundle)
{
	// Retrieve the regression signature from meta graph def.
	const auto& signature_def = bundle.GetSignatures().at("serving_default");
	const string input_name = signature_def.inputs().at("normm_input").name();
	const string output_name = signature_def.outputs().at("Out").name();

	std::vector<float> serialized = { 25.0,	55.7,	18.92,	20.38,	28.08,	1.37,	10546.0,	670.0,	10000.0,	0.0,	0.0,	0.0,	0.0,	0.57,	0.77,	1.07,	2.57,	1.82,	1.82,	1.54,	7.3,	1.84,	1.82,	2.59,	6.29,	1.54,	3.23,	4.19,	5.46,	3.44,	1.82,	3.69,	5.45,	1.82,	1.82,	2.59,	8.13,	1.54,	2.07,	2.55,	5.46,	1.78,	1.79,	1.82,	7.48,	1.84,	1.82,	3.05,	5.78,	1.92,	1.87,	3.09,	8.69,	2.32,	1.89,	4.15,	9.67,	3.09,	5.73,	3.09,	8.47,	3.13,	3.27,	3.45,	2.53,	3.28,	2.7,	0.77,	8.53,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	1.8,	1.82,	1.54,	7.3,	1.84,	1.82,	2.59,	6.29,	1.54,	3.22,	4.2,	5.48,	1.82,	1.82,	2.59,	8.13,	1.54,	2.07,	2.55,	5.46,	1.78,	1.79,	1.82,	7.48,	1.84,	1.82,	3.05,	5.78,	2.0,	2.49,	3.41,	5.26,	1.92,	1.87,	3.09,	8.71,	2.32,	1.89,	4.15,	9.66,	3.09,	5.73,	1.52,	10.79,	3.15,	3.45,	3.85,	9.0,	2.53,	0.75,	3.47,	6.23,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 };

	std::vector<float> serialized_examples;
	for (size_t i = 0; i < serialized.size(); ++i) {
		std::cout << serialized[i] << " ";
		serialized_examples.push_back(serialized[i]);
	}

	/* Here tested my model. Compared results of before exported and here results. it must be same result with same inputs.*/
	std::vector<Tensor> outputs;
	for (size_t i = 0; i < 129; i++)
	{
		serialized_examples[9 + i] = serialized_examples[9 + i] + 1;
		Tensor input = test::AsTensor<float>(serialized_examples, TensorShape({ 150 }));
		bundle.session->Run({ {input_name, input} }, { output_name }, {}, &outputs);
		float* data = static_cast<float*>(outputs[0].data());
		std::cout <<i<< " TTTTTTTTTTTTTTTTTTTT " << *data << std::endl;

		serialized_examples.clear();
		for (size_t i = 0; i < serialized.size(); ++i) {
			//std::cout << serialized[i] << " ";
			serialized_examples.push_back(serialized[i]);
		}
	}
}


 
