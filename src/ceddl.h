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

using namespace eddll;

extern "C" {

    typedef void* tensor;
    typedef void* layer;
    typedef void* model;
    typedef void* optimizer;
    typedef void* callback;
    typedef void* initializer;
    typedef void* loss;
    typedef void* metric;
    typedef void* compserv;

    // ---- TENSOR ----
    tensor ceddl_tensor(const int* shape, int shape_count, float *ptr);

    tensor ceddl_tensor_load(const char* fname);

    float *ceddl_tensor_getptr(tensor t);

    // ---- TENSOR OPERATIONS ----
    void ceddl_div(tensor t, float v);

    // ---- CORE LAYERS ----
    layer ceddl_Activation(layer parent, char* activation, char* name);

    layer ceddl_Conv(layer parent, int filters, const int* kernel_size, int kernel_size_count,
            const int* strides, int strides_count, const char* padding, int groups,
            const int* dilation_rate, int dilation_rate_count,
            bool use_bias, const char* name);
    layer ceddl_ConvT(layer parent, int filters, const int* kernel_size, int kernel_size_count,
            const int* output_padding, int output_padding_count, const char* padding,
            const int* dilation_rate, int dilation_rate_count,
            const int* strides, int strides_count,
            bool use_bias, const char* name);
    layer ceddl_Dense(layer parent, int ndim, bool use_bias, const char* name);
    layer ceddl_Embedding(int input_dim, int output_dim, const char* name);
    layer ceddl_Input(const int* shape, int shape_count, const char* name);

    layer ceddl_UpSampling(layer parent, const int* size, int size_count, const char* interpolation,
            const char* name);
    layer ceddl_Reshape(layer parent, const int* shape, int shape_count, const char* name);

    layer ceddl_Transpose(layer parent, const int* dims, int dims_count, const char* name);

    // ---- LOSSES ----
    loss ceddl_getLoss(const char* type);

    // ---- METRICS ----
    metric ceddl_getMetric(const char* type);

    // ---- MERGE LAYERS ----
    layer ceddl_Add(const layer* layers, int layers_count, const char* name);

    layer ceddl_Average(const layer* layers, int layers_count, const char* name);
    layer ceddl_Concat(const layer* layers, int layers_count, const char* name);

    layer ceddl_MatMul(const layer* layers, int layers_count, const char* name);
    layer ceddl_Maximum(const layer* layers, int layers_count, const char* name);
    layer ceddl_Minimum(const layer* layers, int layers_count, const char* name);
    layer ceddl_Subtract(const layer* layers, int layers_count, const char* name);

    // ---- NOISE LAYERS ----
    layer ceddl_GaussianNoise(layer parent, float stddev, const char* name = ""); //Todo: Implement

    // ---- NORMALIZATION LAYERS ----
    layer ceddl_BatchNormalization(layer parent, float momentum, float epsilon, bool affine, const char* name);
    layer ceddl_Dropout(layer parent, float rate, const char* name = "");

    // ---- OPERATOR LAYERS ----
    layer ceddl_Abs(layer l);

    layer ceddl_Diff(layer l1, layer l2);

    layer ceddl_Diff_scalar(layer l1, float k);

    layer ceddl_Div(layer l1, layer l2);

    layer ceddl_Div_scalar(layer l1, float k);

    layer ceddl_Exp(layer l);

    layer ceddl_Log(layer l);

    layer ceddl_Log2(layer l);

    layer ceddl_Log10(layer l);

    layer ceddl_Mult(layer l1, layer l2);

    layer ceddl_Mult_scalar(layer l1, float k);

    layer ceddl_Pow(layer l1, layer l2);

    layer ceddl_Pow_scalar(layer l1, float k);

    layer ceddl_Sqrt(layer l);

    layer ceddl_Sum(layer l1, layer l2);

    layer ceddl_Sum_scalar(layer l1, float k);

    // ---- REDUCTION LAYERS ----
    layer ceddl_ReduceMean(layer l, const int* axis, int axis_count, bool keepdims = false);

    layer ceddl_ReduceVar(layer l, const int* axis, int axis_count, bool keepdims = false);

    layer ceddl_ReduceSum(layer l, const int* axis, int axis_count, bool keepdims = false);

    layer ceddl_ReduceMax(layer l, const int* axis, int axis_count, bool keepdims = false);

    layer ceddl_ReduceMin(layer l, const int* axis, int axis_count, bool keepdims = false);

    // ---- GENERATOR LAYERS ----
    layer ceddl_GaussGenerator(float mean, float stdev, const int* size, int size_count);

    layer ceddl_UniformGenerator(float low, float high, const int* size, int size_count);

    // ---- OPTIMIZERS ----
    optimizer ceddl_adadelta(float lr, float rho, float epsilon, float weight_decay);
    optimizer ceddl_adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay,
        bool amsgrad);
    optimizer ceddl_adagrad(float lr, float epsilon, float weight_decay);
    optimizer ceddl_adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay);
    optimizer ceddl_nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay);
    optimizer ceddl_rmsprop(float lr, float rho, float epsilon, float weight_decay);
    optimizer ceddl_sgd(float lr, float momentum, float weight_decay, bool nesterov);


    // ---- POOLING LAYERS ----
    layer ceddl_AveragePool(layer parent,
        const int* pool_size, int pool_size_count,
        const int* strides, int strides_count,
        const char* padding, const char* name);

    layer ceddl_GlobalMaxPool(layer parent, const char* name);
    layer ceddl_GlobalAveragePool(layer parent, const char* name);
    layer ceddl_MaxPool(layer parent,
        const int* pool_size, int pool_size_count,
        const int* strides, int strides_count,
        const char* padding, const char* name);


    // ---- RECURRENT LAYERS ----
    layer ceddl_RNN(layer parent, int units, int num_layers, bool use_bias, float dropout,
        bool bidirectional, const char* name);

    layer ceddl_LSTM(layer parent, int units, int num_layers, bool use_bias, float dropout,
        bool bidirectional, const char* name);


    //    // ---- LR SCHEDULERS ----
    //    callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch); //Todo: Implement
    //    callback ExponentialLR(float gamma, int last_epoch); //Todo: Implement
    //    callback MultiStepLR(const vector<int> &milestones, float gamma, int last_epoch); //Todo: Implement
    //    callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps); //Todo: Implement
    //    callback StepLR(int step_size, float gamma, int last_epoch); //Todo: Implement

    // ---- INITIALIZERS ----
    initializer ceddl_Constant(float value);
    initializer ceddl_Identity(float gain);
    initializer ceddl_GlorotNormal(float seed);
    initializer ceddl_GlorotUniform(float seed);
    initializer ceddl_RandomNormal(float mean, float stdev, int seed);
    initializer ceddl_RandomUniform(float minval, float maxval, int seed);
    initializer ceddl_Orthogonal(float gain, int seed);


    // ---- COMPUTING SERVICES ----
    compserv ceddl_CS_CPU(int th, int lsb);

    compserv ceddl_CS_GPU(const int* g, int g_count, int lsb);

    compserv ceddl_CS_FGPA(const int* f, int f_count, int lsb);

    compserv ceddl_CS_COMPSS(char* path);


    // ---- MODEL METHODS ----
    model ceddl_Model(layer in, layer out);

    void ceddl_build(model net, optimizer o, const char** lo, int lo_count, const char** me, int me_count, compserv cs);

    const char* ceddl_summary(model m);

    void ceddl_load(model m, const char* fname);

    void ceddl_save(model m, const char* fname);

    void ceddl_plot(model m, const char* fname);

    void ceddl_fit(model m, const tensor* in, int in_count, const tensor* out, int out_count, int batch, int epochs);

    void ceddl_evaluate(model m, const tensor* in, int in_count, const tensor* out, int out_count);

    void ceddl_predict(model m, const tensor* in, int in_count, const tensor* out, int out_count);

    model ceddl_load_model(const char* fname);
    void ceddl_save_model(model m, const char* fname);
    void ceddl_set_trainable(model m);
    model ceddl_zoo_models(const char* model_name);
    bool ceddl_exist(const char* name);

    // ---- LAYER METHODS ----
    void ceddl_set_trainable(layer l);
    layer ceddl_get_layer(model m, const char* layer_name);


    // ---- DATASETS ----
    void ceddl_download_mnist();

    // ---- MODELS FOR TESTING ----
    model ceddl_get_model_mlp(int batch_size);

    model ceddl_get_model_cnn(int batch_size);

}