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
	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor(const int* shape, int shape_count, float *ptr);

	CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor_load(const char* fname);

	CEDDLL_API float* CALLING_CONV ceddl_tensor_getptr(tensor_ptr t);

    // ---- TENSOR OPERATIONS ----
	CEDDLL_API void CALLING_CONV ceddl_div(tensor_ptr t, float v);
	CEDDLL_API int CALLING_CONV ceddl_ndim(tensor_ptr t);
	CEDDLL_API int CALLING_CONV ceddl_size(tensor_ptr t);
	CEDDLL_API int CALLING_CONV ceddl_info(tensor_ptr t);

    // ---- CORE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Activation(layer_ptr parent, char* activation, float param, char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_ReLu(layer_ptr parent);
	
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Conv(layer_ptr parent, int filters, const int* kernel_size, int kernel_size_count,
            const int* strides, int strides_count, const char* padding, int groups,
            const int* dilation_rate, int dilation_rate_count,
            bool use_bias, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_ConvT(layer_ptr parent, int filters, const int* kernel_size, int kernel_size_count,
            const int* output_padding, int output_padding_count, const char* padding,
            const int* dilation_rate, int dilation_rate_count,
            const int* strides, int strides_count,
            bool use_bias, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dense(layer_ptr parent, int num_dim, bool use_bias, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Embedding(int input_dim, int output_dim, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Input(const int* shape, int shape_count, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_UpSampling(layer_ptr parent, const int* size, int size_count, const char* interpolation,
            const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Reshape(layer_ptr parent, const int* shape, int shape_count, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Transpose(layer_ptr parent, const int* dims, int dims_count, const char* name);

    // ---- LOSSES ----
	CEDDLL_API loss_ptr CALLING_CONV ceddl_getLoss(const char* type);

    // ---- METRICS ----
	CEDDLL_API metric_ptr CALLING_CONV ceddl_getMetric(const char* type);

    // ---- MERGE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Add(const layer_ptr* layers, int layers_count, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_Average(const layer_ptr* layers, int layers_count, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Concat(const layer_ptr* layers, int layers_count, const char* name);

	CEDDLL_API layer_ptr CALLING_CONV ceddl_MatMul(const layer_ptr* layers, int layers_count, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Maximum(const layer_ptr* layers, int layers_count, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Minimum(const layer_ptr* layers, int layers_count, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Subtract(const layer_ptr* layers, int layers_count, const char* name);

    // ---- NOISE LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GaussianNoise(layer_ptr parent, float std_dev, const char* name);

    // ---- NORMALIZATION LAYERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_BatchNormalization(layer_ptr parent, float momentum, float epsilon, bool affine, const char* name);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Dropout(layer_ptr parent, float rate, const char* name);

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
	CEDDLL_API layer_ptr CALLING_CONV ceddl_AveragePool(layer_ptr parent,
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
    //    callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch);
    //    callback ExponentialLR(float gamma, int last_epoch);
    //    callback MultiStepLR(const vector<int> &milestones, float gamma, int last_epoch);
    //    callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps);
    //    callback StepLR(int step_size, float gamma, int last_epoch);

    // ---- INITIALIZERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_Constant(layer_ptr parent, float value);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlorotNormal(layer_ptr parent, float seed);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_GlorotUniform(layer_ptr parent, float seed);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RandomNormal(layer_ptr parent, float mean, float std_dev, int seed);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RandomUniform(layer_ptr parent, float min_val, float max_val, int seed);


    // ---- COMPUTING SERVICES ----
	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_CPU(int th);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_GPU(const int* g, int g_count, int lsb);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_FGPA(const int* f, int f_count, int lsb);

	CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_COMPSS(char* path);


    // ---- MODEL METHODS ----
	CEDDLL_API model_ptr CALLING_CONV ceddl_Model(layer_ptr* in, int in_count, layer_ptr* out, int out_count);

	CEDDLL_API void CALLING_CONV ceddl_build(model_ptr net, optimizer_ptr o,
		const char** lo, int lo_count,
		const char** me, int me_count,
		compserv_ptr cs);

	CEDDLL_API void CALLING_CONV ceddl_summary(model_ptr m);

	CEDDLL_API void CALLING_CONV ceddl_load(model_ptr m, const char* fname);

	CEDDLL_API void CALLING_CONV ceddl_save(model_ptr m, const char* fname);

	CEDDLL_API void CALLING_CONV ceddl_plot(model_ptr m, const char* fname);

	CEDDLL_API void CALLING_CONV ceddl_fit(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count,
		int batch, int epochs);

	CEDDLL_API void CALLING_CONV ceddl_evaluate(model_ptr m,
		const tensor_ptr* in, int in_count,
		const tensor_ptr* out, int out_count);

    // ---- DATA SETS ----
	CEDDLL_API void CALLING_CONV ceddl_download_mnist();
	CEDDLL_API void CALLING_CONV ceddl_download_cifar10();

	// ---- REGULARIZERS ----
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL1(layer_ptr parent, float factor);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL2(layer_ptr parent, float factor);
	CEDDLL_API layer_ptr CALLING_CONV ceddl_RegularizerL1L2(layer_ptr parent, float l1_factor, float l2_factor);
}