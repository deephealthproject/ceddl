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

#if defined(_WIN32)

#define CEDDLL_API __declspec(dllexport)
#else
#define CEDDLL_API
#endif

#define CALLING_CONV

extern "C" {

    typedef void* tensor_ptr;
    typedef void* layer_ptr;
    typedef void* model_ptr;
    typedef void* optimizer_ptr;
    typedef void* callback_ptr;
    typedef void* loss_ptr;
    typedef void* metric_ptr;
    typedef void* compserv_ptr;

    // ---- TENSOR ----
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor(const int* shape, int shape_count, float *data);

	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor_load(const char* fname);

	CEDDLL_API float* CALLING_CONV ceddl_tensor_getptr(tensor_ptr t);

    // ---- TENSOR OPERATIONS ----
	CEDDLL_API void CALLING_CONV ceddl_div(tensor_ptr t, float v);
	CEDDLL_API int CALLING_CONV ceddl_ndim(tensor_ptr t);
	CEDDLL_API int CALLING_CONV ceddl_size(tensor_ptr t);
	CEDDLL_API void CALLING_CONV ceddl_print(tensor_ptr t);
	CEDDLL_API void CALLING_CONV ceddl_info(tensor_ptr t);
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_select(tensor_ptr t, const char** indices, int indices_count);

    ///////////////////////////////////////
    //  MODEL METHODS
    ///////////////////////////////////////
	
	// Creation
	CEDDLL_API model_ptr CALLING_CONV ceddl_Model(
		layer_ptr* in, int in_count, const char** in_types,
		layer_ptr* out, int out_count, const char** out_types
	);

	CEDDLL_API void CALLING_CONV ceddl_build(
		model_ptr net,
		optimizer_ptr o, const char* type,
		const char** lo, int lo_count,
		const char** me, int me_count,
		compserv_ptr cs
	);
	
	// Computing services
	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_CPU(int th);
	
	// Info and logs
	CEDDLL_API void CALLING_CONV ceddl_setlogfile(model_ptr m, const char* fname);
	CEDDLL_API void CALLING_CONV ceddl_summary(model_ptr m);
	CEDDLL_API void CALLING_CONV ceddl_plot(model_ptr m, const char* fname);

	// Serialization
	CEDDLL_API void CALLING_CONV ceddl_load(model_ptr m, const char* fname);
	CEDDLL_API void CALLING_CONV ceddl_save(model_ptr m, const char* fname);

	// Optimizer
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adadelta(float lr, float rho, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay,
		bool amsgrad);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adagrad(float lr, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_rmsprop(float lr, float rho, float epsilon, float weight_decay);
	CEDDLL_API optimizer_ptr CALLING_CONV ceddl_sgd(float lr, float momentum, float weight_decay, bool nesterov);
	
	// Training and Evaluation
	CEDDLL_API void CALLING_CONV ceddl_fit(
		model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count,
		int batch, int epochs
	);
	CEDDLL_API void CALLING_CONV ceddl_evaluate(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count
	);
	CEDDLL_API void CALLING_CONV ceddl_forward(model_ptr m, const tensor_ptr* in, int in_count);

	// loss and metrics methods
	CEDDLL_API metric_ptr CALLING_CONV ceddl_getMetric(const char* type);
	CEDDLL_API float CALLING_CONV ceddl_getMetricValue(metric_ptr metric, const char* type, tensor_ptr tensorT, tensor_ptr tensorY);
	
	///////////////////////////////////////
    //  LAYERS
    ///////////////////////////////////////

	// Core Layers
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Activation(
		layer_ptr parent, const char* parent_type,
		char* activation,
		float* params, int params_size,
		char* name
    );
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Sigmoid(layer_ptr parent, const char* parent_type, char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Softmax(layer_ptr parent, const char* parent_type, char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReLu(layer_ptr parent, const char* parent_type);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_LeakyReLu(layer_ptr parent, const char* parent_type);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Conv(
		layer_ptr parent, const char* parent_type,
		int filters,
		const int* kernel_size, int kernel_size_count,
		const int* strides, int strides_count,
		const char* padding, int groups,
		const int* dilation_rate, int dilation_rate_count,
		bool use_bias, const char* name
	);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dense(
		layer_ptr parent, const char* parent_type,
		int num_dim,
		bool use_bias,
		const char* name
	);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Input(const int* shape, int shape_count, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_UpSampling(
		layer_ptr parent, const char* parent_type,
		const int* size, int size_count,
		const char* interpolation,
		const char* name
	);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Reshape(
		layer_ptr parent, const char* parent_type,
		const int* shape, int shape_count,
		const char* name
	);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Flatten(layer_ptr parent, const char* parent_type, const char* name);

	// ---- MERGE LAYERS ----
	
	CEDDLL_API model_ptr CALLING_CONV ceddl_Concat(
		layer_ptr* in, int in_count, const char** in_types
	);

	// ---- NOISE LAYERS ----

	// ---- NORMALIZATION LAYERS ----
	
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_BatchNormalization(layer_ptr layer, const char* layer_type);

	// ---- OPERATOR LAYERS ----

	// ---- REDUCTION LAYERS ----

	// ---- GENERATOR LAYERS ----

	// ---- POOLING LAYERS ----

	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalMaxPool(layer_ptr parent, const char* parent_type, const char* name);
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_MaxPool(
		layer_ptr parent, const char* parent_type,
		const int* pool_size, int pool_size_count,
		const int* strides, int strides_count,
		const char* padding, const char* name
	);

	// Recurrent Layers

    //////////////////////////////
    // Layers Methods
    //////////////////////////////

	////////////////////////////////////
    // Manage Tensors inside Layers
    ////////////////////////////////////

	CEDDLL_API tensor_ptr CALLING_CONV ceddl_GetOutput(layer_ptr layer, const char* layer_type);

	///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////

    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////

    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
	
	CEDDLL_API void CALLING_CONV ceddl_download_mnist();
	CEDDLL_API void CALLING_CONV ceddl_download_cifar10();
}