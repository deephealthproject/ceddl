/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Ynse Hoornenborg: ynse.hoornenborg@philips.com
//
//
// To collaborate please contact ynse.hoornenborg@philips.com
//
/////////////////////////////////////////////////////////////////////////////

#include <ceddl.h>
#include <eddl.h>

template <class T>
void fillVector(const std::vector<T> &vector, const T* arr, int arr_count) {
	for (int i = 0; i < arr_count; i++) {
		vector.push_back(arr[i]);
	}
}

void fillStringsVector(const std::vector<string> &vector, const char** arr, int arr_count) {
	for (int i = 0; i < arr_count; i++) {
		vector.push_back(string(arr[i]));
	}
}

void fillLayersVector(const std::vector<layer> &vector, const layer_ptr* arr, int arr_count) {
	for (int i = 0; i < arr_count; i++) {
		layer const temp = static_cast<layer>(arr[i]);
		vector.push_back(temp);
	}
}

void fillTensorsVector(const std::vector<tensor> &vector, const tensor_ptr* arr, int arr_count) {
	for (int i = 0; i < arr_count; i++) {
		layer const temp = static_cast<tensor>(arr[i]);
		vector.push_back(temp);
	}
}

extern "C" {


	// ---- TENSOR ----
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor(const int* shape, int shape_count, float *ptr) {
		const std::vector<int> shape_vector = std::vector<int>();
		fillVector(shape_vector, shape, shape_count);
		return eddl::T(shape_vector, ptr);
	}

	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor_load(const char* fname) {
		const std::string filename = string(fname);
		return eddl::T_load(filename);
	}

	CEDDLL_API float* CALLING_CONV ceddl_tensor_getptr(tensor_ptr t) {
		return eddl::T_getptr(static_cast<tensor>(t));
	}

	// ---- TENSOR OPERATIONS ----
	CEDDLL_API void CALLING_CONV ceddl_div(tensor_ptr t, float v) {
		eddl::div(static_cast<tensor>(t), v);
	}

	// ---- CORE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Activation(layer_ptr parent, char* activation, char* name) {
		const std::string activation_str = string(activation);
		const std::string name_str = string(name);
		return eddl::Activation(static_cast<layer>(parent), activation_str, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Conv(layer_ptr parent, int filters,
		const int* kernel_size, int kernel_size_count,
		const int* strides, int strides_count,
		const char* padding, int groups,
		const int* dilation_rate, int dilation_rate_count,
		bool use_bias, const char* name
	) {
		const std::vector<int> kernel_size_vector = std::vector<int>();
		const std::vector<int> strides_vector = std::vector<int>();
		const std::vector<int> dilation_rate_vector = std::vector<int>();
		const std::string name_str = string(name);
		const std::string padding_str = string(padding);
		fillVector(kernel_size_vector, kernel_size, kernel_size_count);
		fillVector(strides_vector, strides, strides_count);
		fillVector(dilation_rate_vector, dilation_rate, dilation_rate_count);
		return eddl::Conv(
			static_cast<layer>(parent), filters,
			kernel_size_vector, strides_vector, padding_str, groups,
			dilation_rate_vector, use_bias, name_str
		);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ConvT(layer_ptr parent, int filters,
		const int* kernel_size, int kernel_size_count,
		const int* output_padding, int output_padding_count,
		const char* padding,
		const int* dilation_rate, int dilation_rate_count,
		const int* strides, int strides_count,
		bool use_bias, const char* name
	) {
		const std::vector<int> kernel_size_vector = std::vector<int>();
		const std::vector<int> strides_vector = std::vector<int>();
		const std::vector<int> output_padding_vector = std::vector<int>();
		const std::vector<int> dilation_rate_vector = std::vector<int>();
		const std::string name_str = string(name);
		const std::string padding_str = string(padding);
		fillVector(kernel_size_vector, kernel_size, kernel_size_count);
		fillVector(strides_vector, strides, strides_count);
		fillVector(output_padding_vector, output_padding, output_padding_count);
		fillVector(dilation_rate_vector, dilation_rate, dilation_rate_count);
		return eddl::ConvT(
			static_cast<layer>(parent), filters,
			kernel_size_vector, output_padding_vector, padding_str,
			dilation_rate_vector, strides_vector, use_bias, name_str
		);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dense(layer_ptr parent, int num_dim, bool use_bias, const char* name) {
		const std::string name_str = string(name);
		return eddl::Dense(static_cast<layer>(parent), num_dim, use_bias, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Embedding(int input_dim, int output_dim, const char* name) {
		const std::string name_str = string(name);
		return eddl::Embedding(input_dim, output_dim, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Input(const int* shape, int shape_count, const char* name) {
		const std::vector<int> shape_vector = std::vector<int>();
		fillVector(shape_vector, shape, shape_count);
		const std::string name_str = string(name);
		return eddl::Input(shape_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_UpSampling(layer parent, const int* size, int size_count, const char* interpolation,
		const char* name
	) {
		const std::vector<int> size_vector = std::vector<int>();
		fillVector(size_vector, size, size_count);
		const std::string name_str = string(name);
		const std::string interpolation_str = string(interpolation);
		return eddl::UpSampling(static_cast<layer>(parent), size_vector, interpolation_str, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Reshape(layer parent, const int* shape, int shape_count, const char* name) {
		const std::vector<int> shape_vector = std::vector<int>();
		fillVector(shape_vector, shape, shape_count);
		const std::string name_str = string(name);
		return eddl::Reshape(static_cast<layer>(parent), shape_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Transpose(layer parent, const int* dims, int dims_count, const char* name) {
		const std::vector<int> dims_vector = std::vector<int>();
		fillVector(dims_vector, dims, dims_count);
		const std::string name_str = string(name);
		return eddl::Transpose(static_cast<layer>(parent), dims_vector, name_str);
	}

	// ---- LOSSES ----
	CEDDLL_API loss_ptr CALLING_CONV ceddl_getLoss(const char* type) {
		const std::string type_str = string(type);
		return eddl::getLoss(type_str);
	}

	// ---- METRICS ----
	CEDDLL_API metric_ptr CALLING_CONV ceddl_getMetric(const char* type) {
		const std::string type_str = string(type);
		return eddl::getMetric(type_str);
	}

	// ---- MERGE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Add(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Add(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Average(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Average(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Concat(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Concat(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_MatMul(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::MatMul(layers_vector, name_str);
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Maximum(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Maximum(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Minimum(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Minimum(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Subtract(const layer_ptr* layers, int layers_count, const char* name) {
		const std::vector<layer> layers_vector = std::vector<layer>();
		fillLayersVector(layers_vector, layers, layers_count);
		const std::string name_str = string(name);
		return eddl::Subtract(layers_vector, name_str);
	}
	
	// ---- NOISE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GaussianNoise(layer_ptr parent, float std_dev, const char* name) {
		const std::string name_str = string(name);
		return eddl::GaussianNoise(static_cast<layer>(parent), std_dev, name_str);
	}

	// ---- NORMALIZATION LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_BatchNormalization(layer_ptr parent, float momentum, float epsilon, bool affine, const char* name) {
		const std::string name_str = string(name);
		return eddl::BatchNormalization(static_cast<layer>(parent), momentum, epsilon, affine, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dropout(layer_ptr parent, float rate, const char* name) {
		const std::string name_str = string(name);
		return eddl::Dropout(static_cast<layer>(parent), rate, name_str);
	}

	// ---- OPERATOR LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Abs(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Diff(layer_ptr l1, layer_ptr l2);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Diff_scalar(layer_ptr l1, float k);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Div(layer_ptr l1, layer_ptr l2);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Div_scalar(layer_ptr l1, float k);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Exp(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log2(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log10(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Mult(layer_ptr l1, layer_ptr l2);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Mult_scalar(layer_ptr l1, float k);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Pow(layer_ptr l1, layer_ptr l2);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Pow_scalar(layer_ptr l1, float k);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sqrt(layer_ptr l);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sum(layer_ptr l1, layer_ptr l2);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sum_scalar(layer_ptr l1, float k);

	// ---- REDUCTION LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMean(layer_ptr l, const int* axis, int axis_count, bool keep_dims);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceVar(layer_ptr l, const int* axis, int axis_count, bool keep_dims);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceSum(layer_ptr l, const int* axis, int axis_count, bool keep_dims);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMax(layer_ptr l, const int* axis, int axis_count, bool keep_dims);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMin(layer_ptr l, const int* axis, int axis_count, bool keep_dims);

	// ---- GENERATOR LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GaussGenerator(float mean, float std_dev, const int* size, int size_count);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_UniformGenerator(float low, float high, const int* size, int size_count);

	// ---- OPTIMIZERS ----
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adadelta(float lr, float rho, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay,
		bool amsgrad);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adagrad(float lr, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_rmsprop(float lr, float rho, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_sgd(float lr, float momentum, float weight_decay, bool nesterov);


	// ---- POOLING LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_AveragePool(layer parent,
		const int* pool_size, int pool_size_count,
		const int* strides, int strides_count,
		const char* padding, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalMaxPool(layer_ptr parent, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalAveragePool(layer_ptr parent, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_MaxPool(layer_ptr parent,
		const int* pool_size, int pool_size_count,
		const int* strides, int strides_count,
		const char* padding, const char* name);


	// ---- RECURRENT LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RNN(layer_ptr parent, int units, int num_layers, bool use_bias, float dropout,
		bool bidirectional, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_LSTM(layer_ptr parent, int units, int num_layers, bool use_bias, float dropout,
		bool bidirectional, const char* name);


	//    // ---- LR SCHEDULERS ----
	//    callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch); //Todo: Implement
	//    callback ExponentialLR(float gamma, int last_epoch); //Todo: Implement
	//    callback MultiStepLR(const vector<int> &milestones, float gamma, int last_epoch); //Todo: Implement
	//    callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps); //Todo: Implement
	//    callback StepLR(int step_size, float gamma, int last_epoch); //Todo: Implement

	// ---- INITIALIZERS ----
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_Constant(float value);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_Identity(float gain);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_GlorotNormal(float seed);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_GlorotUniform(float seed);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_RandomNormal(float mean, float std_dev, int seed);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_RandomUniform(float min_val, float max_val, int seed);
	CEDDLL_API initializer_ptr CALLING_CONV ceddl_Orthogonal(float gain, int seed);


	// ---- COMPUTING SERVICES ----
	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_CPU(int th, int lsb);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_GPU(const int* g, int g_count, int lsb);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_FGPA(const int* f, int f_count, int lsb);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_COMPSS(char* path);


	// ---- MODEL METHODS ----
	CEDDLL_API model_ptr CALLING_CONV ceddl_Model(layer_ptr* in, int in_count, layer_ptr* out, int out_count) {
		const std::vector<layer> in_vector = std::vector<layer>();
		const std::vector<layer> out_vector = std::vector<layer>();
		fillLayersVector(in_vector, in, in_count);
		fillLayersVector(out_vector, out, out_count);
		return eddl::Model(in_vector, out_vector);
	}

	CEDDLL_API void CALLING_CONV ceddl_build(model_ptr net, optimizer_ptr o, const char** lo, int lo_count, const char** me, int me_count, compserv_ptr cs) {
		const std::vector<string> lo_vector = std::vector<string>();
		const std::vector<string> me_vector = std::vector<string>();
		fillStringsVector(lo_vector, lo, lo_count);
		fillStringsVector(me_vector, me, me_count);
		eddl::build(static_cast<model>(net), static_cast<optimizer>(o), lo_vector, me_vector, static_cast<compserv>(cs));
	}

	CEDDLL_API const char* CALLING_CONV ceddl_summary(model_ptr m) {
		string summary = eddl::summary(static_cast<model>(m));
		return summary.c_str();
	}

	CEDDLL_API void CALLING_CONV ceddl_load(model_ptr m, const char* fname) {
		const std::string fname_str = string(fname);
		eddl::load(static_cast<model>(m), fname_str);
	}

	CEDDLL_API void CALLING_CONV ceddl_save(model_ptr m, const char* fname) {
		const std::string fname_str = string(fname);
		eddl::save(static_cast<model>(m), fname_str);
	}

	CEDDLL_API void CALLING_CONV ceddl_plot(model_ptr m, const char* fname) {
		const std::string fname_str = string(fname);
		eddl::plot(static_cast<model>(m), fname_str);
	}

	CEDDLL_API void CALLING_CONV ceddl_fit(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count,
		int batch, int epochs
	) {
		const std::vector<tensor> in_vector = std::vector<tensor>();
		const std::vector<tensor> out_vector = std::vector<tensor>();
		fillTensorsVector(in_vector, in, in_count);
		fillTensorsVector(out_vector, out, out_count);
		eddl::fit(static_cast<model>(m), in_vector, out_vector, batch, epochs);
	}

	CEDDLL_API void CALLING_CONV ceddl_evaluate(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count
	) {
		const std::vector<tensor> in_vector = std::vector<tensor>();
		const std::vector<tensor> out_vector = std::vector<tensor>();
		fillTensorsVector(in_vector, in, in_count);
		fillTensorsVector(out_vector, out, out_count);
		eddl::evaluate(static_cast<model>(m), in_vector, out_vector);
	}

	CEDDLL_API void CALLING_CONV ceddl_predict(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count
	) {
		const std::vector<tensor> in_vector = std::vector<tensor>();
		const std::vector<tensor> out_vector = std::vector<tensor>();
		fillTensorsVector(in_vector, in, in_count);
		fillTensorsVector(out_vector, out, out_count);
		eddl::predict(static_cast<model>(m), in_vector, out_vector);
	}

	CEDDLL_API model_ptr CALLING_CONV ceddl_load_model(const char* fname) {
		const std::string fname_str = string(fname);
		return eddl::load_model(fname_str);
	}

	CEDDLL_API void CALLING_CONV ceddl_save_model(model_ptr m, const char* fname) {
		const std::string fname_str = string(fname);
		eddl::save_model(static_cast<model>(m), fname_str);
	}

	CEDDLL_API void CALLING_CONV ceddl_set_trainable(model_ptr m) {
		eddl::set_trainable(static_cast<model>(m));
	}

	CEDDLL_API model_ptr CALLING_CONV ceddl_zoo_models(const char* model_name) {
		const std::string model_name_str = string(model_name);
		return eddl::zoo_models(model_name_str);
	}

	CEDDLL_API bool CALLING_CONV ceddl_exist(const char* name) {
		const std::string name_str = string(name);
		return eddl::exist(name_str);
	}

	// ---- LAYER METHODS ----
	CEDDLL_API void CALLING_CONV ceddl_set_trainable_layer(layer_ptr l) {
		eddl::set_trainable(static_cast<layer>(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_get_layer(model_ptr m, const char* layer_name) {
		const std::string layer_name_str = string(layer_name);
		return eddl::get_layer(static_cast<model>(m), layer_name_str);
	}


	// ---- DATA SETS ----
	CEDDLL_API void CALLING_CONV ceddl_download_mnist() {
		eddl::download_mnist();
	}

	// ---- MODELS FOR TESTING ----
	CEDDLL_API model_ptr CALLING_CONV ceddl_get_model_mlp(int batch_size) {
		return eddl::get_model_mlp(batch_size);
	}

	CEDDLL_API model_ptr CALLING_CONV ceddl_get_model_cnn(int batch_size) {
		return eddl::get_model_cnn(batch_size);
	}


}