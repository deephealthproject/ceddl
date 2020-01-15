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
#include <eddlT.h>
#include <algorithm>
#include <iostream>

string transformString(const char* s) {
	return string(s);
}

layer transformLayer(layer_ptr l) {
	return static_cast<layer>(l);
}

tensor transformTensor(tensor_ptr t) {
	return static_cast<tensor>(t);
}

template <class T, class T_ptr, class Func>
void fillVector(std::vector<T> &vector, const T_ptr* arr, int arr_count, Func func) {
	std::vector<T_ptr> in_vector(arr, arr + arr_count);
	for(int i = 0; i < in_vector.size(); i++) {
		vector.push_back(func(in_vector[i]));
	}
}

extern "C" {

	// ---- TENSOR ----
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor(const int* shape, int shape_count, float *ptr) {
		const std::vector<int> shape_vector(shape, shape + shape_count);
		return eddlT::create(shape_vector, ptr);
	}

	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor_load(const char* fname) {
		const std::string filename = string(fname);
		return eddlT::load(filename);
	}

	CEDDLL_API float* CALLING_CONV ceddl_tensor_getptr(tensor_ptr t) {
		return eddlT::getptr(transformTensor(t));
	}

	// ---- TENSOR OPERATIONS ----
	CEDDLL_API void CALLING_CONV ceddl_div(tensor_ptr t, float v) {
		eddlT::div_(transformTensor(t), v);
	}

	CEDDLL_API int CALLING_CONV ceddl_ndim(tensor_ptr t) {
		tensor t1 = transformTensor(t);
		return t1->ndim;
	}

	CEDDLL_API int CALLING_CONV ceddl_size(tensor_ptr t) {
		tensor t1 = transformTensor(t);
		return t1->size;
	}

	CEDDLL_API void CALLING_CONV ceddl_info(tensor_ptr t) {
		eddlT::info(transformTensor(t));
	}

	// ---- CORE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Activation(layer_ptr parent, char* activation, float param, char* name) {
		const std::string activation_str = string(activation);
		const std::string name_str = string(name);
		return eddl::Activation(transformLayer(parent), activation_str, param, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReLu(layer_ptr parent) {
		return eddl::ReLu(transformLayer(parent));
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Conv(layer_ptr parent, int filters,
		const int* kernel_size, int kernel_size_count,
		const int* strides, int strides_count,
		const char* padding, int groups,
		const int* dilation_rate, int dilation_rate_count,
		bool use_bias, const char* name
	) {
		const std::vector<int> kernel_size_vector(kernel_size, kernel_size + kernel_size_count);
		const std::vector<int> strides_vector(strides, strides + strides_count);
		const std::vector<int> dilation_rate_vector(dilation_rate, dilation_rate + dilation_rate_count);
		const std::string name_str(name);
		const std::string padding_str(padding);
		return eddl::Conv(
			transformLayer(parent), filters,
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
		const std::vector<int> kernel_size_vector(kernel_size, kernel_size + kernel_size_count);
		const std::vector<int> strides_vector(strides, strides + strides_count);
		const std::vector<int> output_padding_vector(output_padding, output_padding + output_padding_count);
		const std::vector<int> dilation_rate_vector(dilation_rate, dilation_rate + dilation_rate_count);
		const std::string name_str = string(name);
		const std::string padding_str = string(padding);
		return eddl::ConvT(
			transformLayer(parent), filters,
			kernel_size_vector, output_padding_vector, padding_str,
			dilation_rate_vector, strides_vector, use_bias, name_str
		);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dense(layer_ptr parent, int num_dim, bool use_bias, const char* name) {
		const std::string name_str = string(name);
		return eddl::Dense(transformLayer(parent), num_dim, use_bias, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Embedding(int input_dim, int output_dim, const char* name) {
		const std::string name_str = string(name);
		return eddl::Embedding(input_dim, output_dim, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Input(const int* shape, int shape_count, const char* name) {
		const std::vector<int> shape_vector(shape, shape + shape_count);
		const std::string name_str = string(name);
		return eddl::Input(shape_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_UpSampling(layer_ptr parent, const int* size, int size_count, const char* interpolation,
		const char* name
	) {
		const std::vector<int> size_vector(size, size + size_count);
		const std::string name_str = string(name);
		const std::string interpolation_str = string(interpolation);
		return eddl::UpSampling(transformLayer(parent), size_vector, interpolation_str, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Reshape(layer_ptr parent, const int* shape, int shape_count, const char* name) {
		const std::vector<int> shape_vector(shape, shape + shape_count);
		const std::string name_str = string(name);
		return eddl::Reshape(transformLayer(parent), shape_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Transpose(layer_ptr parent, const char* name) {
		const std::string name_str = string(name);
		return eddl::Transpose(transformLayer(parent), name_str);
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
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Add(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Average(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Average(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Concat(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Concat(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_MatMul(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::MatMul(layers_vector, name_str);
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Maximum(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Maximum(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Minimum(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Minimum(layers_vector, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Subtract(const layer_ptr* layers, int layers_count, const char* name) {
		std::vector<layer> layers_vector = std::vector<layer>();
		fillVector(layers_vector, layers, layers_count, transformLayer);
		const std::string name_str = string(name);
		return eddl::Subtract(layers_vector, name_str);
	}
	
	// ---- NOISE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GaussianNoise(layer_ptr parent, float std_dev, const char* name) {
		const std::string name_str = string(name);
		return eddl::GaussianNoise(transformLayer(parent), std_dev, name_str);
	}

	// ---- NORMALIZATION LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_BatchNormalization(layer_ptr parent, float momentum, float epsilon, bool affine, const char* name) {
		const std::string name_str = string(name);
		return eddl::BatchNormalization(transformLayer(parent), momentum, epsilon, affine, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dropout(layer_ptr parent, float rate, const char* name) {
		const std::string name_str = string(name);
		return eddl::Dropout(transformLayer(parent), rate, name_str);
	}

	// ---- OPERATOR LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Abs(layer_ptr l) {
		return eddl::Abs(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Diff(layer_ptr l1, layer_ptr l2) {
		return eddl::Diff(transformLayer(l1), transformLayer(l2));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Diff_scalar(layer_ptr l1, float k) {
		return eddl::Diff(transformLayer(l1), k);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Div(layer_ptr l1, layer_ptr l2) {
		return eddl::Div(transformLayer(l1), transformLayer(l2));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Div_scalar(layer_ptr l1, float k) {
		return eddl::Div(transformLayer(l1), k);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Exp(layer_ptr l) {
		return eddl::Exp(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log(layer_ptr l) {
		return eddl::Log(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log2(layer_ptr l) {
		return eddl::Log2(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Log10(layer_ptr l) {
		return eddl::Log10(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Mult(layer_ptr l1, layer_ptr l2) {
		return eddl::Mult(transformLayer(l1), transformLayer(l2));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Mult_scalar(layer_ptr l1, float k) {
		return eddl::Mult(transformLayer(l1), k);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Pow(layer_ptr l1, layer_ptr l2) {
		return eddl::Pow(transformLayer(l1), transformLayer(l2));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Pow_scalar(layer_ptr l1, float k) {
		return eddl::Pow(transformLayer(l1), k);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sqrt(layer_ptr l) {
		return eddl::Sqrt(transformLayer(l));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sum(layer_ptr l1, layer_ptr l2) {
		return eddl::Sum(transformLayer(l1), transformLayer(l2));
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sum_scalar(layer_ptr l1, float k) {
		return eddl::Sum(transformLayer(l1), k);
	}

	// ---- REDUCTION LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMean(layer_ptr l, const int* axis, int axis_count, bool keep_dims) {
		const std::vector<int> axis_vector(axis, axis + axis_count);
		return eddl::ReduceMean(transformLayer(l), axis_vector, keep_dims);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceVar(layer_ptr l, const int* axis, int axis_count, bool keep_dims) {
		const std::vector<int> axis_vector(axis, axis + axis_count);
		return eddl::ReduceVar(transformLayer(l), axis_vector, keep_dims);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceSum(layer_ptr l, const int* axis, int axis_count, bool keep_dims) {
		const std::vector<int> axis_vector(axis, axis + axis_count);
		return eddl::ReduceSum(transformLayer(l), axis_vector, keep_dims);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMax(layer_ptr l, const int* axis, int axis_count, bool keep_dims) {
		const std::vector<int> axis_vector(axis, axis + axis_count);
		return eddl::ReduceMax(transformLayer(l), axis_vector, keep_dims);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReduceMin(layer_ptr l, const int* axis, int axis_count, bool keep_dims) {
		const std::vector<int> axis_vector(axis, axis + axis_count);
		return eddl::ReduceMin(transformLayer(l), axis_vector, keep_dims);
	}

	// ---- GENERATOR LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GaussGenerator(float mean, float std_dev, const int* size, int size_count) {
		const std::vector<int> size_vector(size, size + size_count);
		return eddl::GaussGenerator(mean, std_dev, size_vector);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_UniformGenerator(float low, float high, const int* size, int size_count) {
		const std::vector<int> size_vector(size, size + size_count);
		return eddl::UniformGenerator(low, high, size_vector);
	}

	// ---- OPTIMIZERS ----
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adadelta(float lr, float rho, float epsilon, float weight_decay) {
		return eddl::adadelta(lr, rho, epsilon, weight_decay);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay,
		bool amsgrad) {
		return eddl::adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adagrad(float lr, float epsilon, float weight_decay) {
		return eddl::adagrad(lr, epsilon, weight_decay);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay) {
		return eddl::adamax(lr, beta_1, beta_2, epsilon, weight_decay);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay) {
		return eddl::nadam(lr, beta_1, beta_2, epsilon, schedule_decay);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_rmsprop(float lr, float rho, float epsilon, float weight_decay) {
		return eddl::rmsprop(lr, rho, epsilon, weight_decay);
	}

	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_sgd(float lr, float momentum, float weight_decay, bool nesterov) {
		return eddl::sgd(lr, momentum, weight_decay, nesterov);
	}


	// ---- POOLING LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_AveragePool(layer_ptr parent,
		const int* pool_size, int pool_size_count,
		const int* strides, int strides_count,
		const char* padding, const char* name
	) {
		const std::vector<int> pool_size_vector(pool_size, pool_size + pool_size_count);
		const std::vector<int> strides_vector(strides, strides + strides_count);
		const std::string padding_str = string(padding);
		const std::string name_str = string(name);
		return eddl::AveragePool(transformLayer(parent), pool_size_vector, strides_vector, padding_str, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalMaxPool(layer_ptr parent, const char* name) {
		const std::string name_str = string(name);
		return eddl::GlobalMaxPool(transformLayer(parent), name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalAveragePool(layer_ptr parent, const char* name) {
		const std::string name_str = string(name);
		return eddl::GlobalAveragePool(transformLayer(parent), name_str);
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_MaxPool(layer_ptr parent,
		const int* pool_size, int pool_size_count,
		const int* strides, int strides_count,
		const char* padding, const char* name
	) {
		const std::vector<int> pool_size_vector(pool_size, pool_size + pool_size_count);
		const std::vector<int> strides_vector(strides, strides + strides_count);
		const std::string padding_str(padding);
		const std::string name_str(name);
		return eddl::MaxPool(transformLayer(parent), pool_size_vector, strides_vector, padding_str, name_str);
	}


	// ---- RECURRENT LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RNN(layer_ptr parent, int units, int num_layers, bool use_bias, float dropout,
		bool bidirectional, const char* name
	) {
		const std::string name_str = string(name);
		return eddl::RNN(transformLayer(parent), units, num_layers, use_bias, dropout, bidirectional, name_str);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_LSTM(layer_ptr parent, int units, int num_layers, bool use_bias, float dropout,
		bool bidirectional, const char* name
	) {
		const std::string name_str = string(name);
		return eddl::LSTM(transformLayer(parent), units, num_layers, use_bias, dropout, bidirectional, name_str);
	}


	//    // ---- LR SCHEDULERS ----
	//    callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch);
	//    callback ExponentialLR(float gamma, int last_epoch);
	//    callback MultiStepLR(const vector<int> &milestones, float gamma, int last_epoch);
	//    callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps);
	//    callback StepLR(int step_size, float gamma, int last_epoch);

	// ---- INITIALIZERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Constant(layer_ptr parent, float value) {
		return eddl::Constant(transformLayer(parent), value);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlorotNormal(layer_ptr parent, float seed) {
		return eddl::GlorotNormal(transformLayer(parent), seed);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlorotUniform(layer_ptr parent, float seed) {
		return eddl::GlorotUniform(transformLayer(parent), seed);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_RandomNormal(layer_ptr parent, float mean, float std_dev, int seed) {
		return eddl::RandomNormal(transformLayer(parent), mean, std_dev, seed);
	}

	CEDDLL_API layer_ptr CALLING_CONV ceddl_RandomUniform(layer_ptr parent, float min_val, float max_val, int seed) {
		return eddl::RandomUniform(transformLayer(parent), min_val, max_val, seed);
	}

	// ---- COMPUTING SERVICES ----
	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_CPU(int th) {
		return eddl::CS_CPU(th);
	}

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_GPU(const int* g, int g_count, int lsb) {
		const std::vector<int> g_vector(g, g + g_count);
		return eddl::CS_GPU(g_vector, lsb);
	}

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_FGPA(const int* f, int f_count, int lsb) {
		const std::vector<int> f_vector(f, f + f_count);
		return eddl::CS_FGPA(f_vector, lsb);
	}

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_COMPSS(char* path) {
		return eddl::CS_COMPSS(path);
	}


	// ---- MODEL METHODS ----
	CEDDLL_API model_ptr CALLING_CONV ceddl_Model(layer_ptr* in, int in_count, layer_ptr* out, int out_count) {
		std::vector<layer> in_vector = std::vector<layer>();
		std::vector<layer> out_vector = std::vector<layer>();
		fillVector(in_vector, in, in_count, transformLayer);
		fillVector(out_vector, out, out_count, transformLayer);
		return eddl::Model(in_vector, out_vector);
	}

	CEDDLL_API void CALLING_CONV ceddl_build(model_ptr net, optimizer_ptr o, const char** lo, int lo_count, const char** me, int me_count, compserv_ptr cs) {
		std::vector<string> lo_vector = std::vector<string>();
		std::vector<string> me_vector = std::vector<string>();
		fillVector(lo_vector, lo, lo_count, transformString);
		fillVector(me_vector, me, me_count, transformString);
		eddl::build(static_cast<model>(net), static_cast<optimizer>(o), lo_vector, me_vector, static_cast<compserv>(cs));
	}

	CEDDLL_API void CALLING_CONV ceddl_summary(model_ptr m) {
		eddl::summary(static_cast<model>(m));
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
		std::vector<tensor> in_vector = std::vector<tensor>();
		std::vector<tensor> out_vector = std::vector<tensor>();
		fillVector(in_vector, in, in_count, transformTensor);
		fillVector(out_vector, out, out_count, transformTensor);
		eddl::fit(static_cast<model>(m), in_vector, out_vector, batch, epochs);
	}

	CEDDLL_API void CALLING_CONV ceddl_evaluate(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count
	) {
		std::vector<tensor> in_vector = std::vector<tensor>();
		std::vector<tensor> out_vector = std::vector<tensor>();
		fillVector(in_vector, in, in_count, transformTensor);
		fillVector(out_vector, out, out_count, transformTensor);
		eddl::evaluate(static_cast<model>(m), in_vector, out_vector);
	}

	// ---- DATA SETS ----
	CEDDLL_API void CALLING_CONV ceddl_download_mnist() {
		eddl::download_mnist();
	}
	
	CEDDLL_API void CALLING_CONV ceddl_download_cifar10() {
		eddl::download_cifar10();
	}
	
	// ---- REGULARIZERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL1(layer_ptr parent, float factor) {
		return eddl::L1(transformLayer(parent), factor);
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL2(layer_ptr parent, float factor) {
		return eddl::L2(transformLayer(parent), factor);
	}
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL1L2(layer_ptr parent, float l1_factor, float l2_factor) {
		return eddl::L1L2(transformLayer(parent), l1_factor, l2_factor);
	}
}