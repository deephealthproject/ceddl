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
#include <algorithm>
#include <iostream>
#include <eddl/tensor/tensor.h>
#include <eddl/metrics/metric.h>
#include <eddl/optimizers/optim.h>
#include <eddl/layers/core/layer_core.h>
#include <eddl/layers/conv/layer_conv.h>
#include <eddl/layers/pool/layer_pool.h>
#include <eddl/serialization/onnx/eddl_onnx.h>

string transformString(const char* s) {
    return string(s);
}

eddl::layer transformLayer(layer_ptr l, string type) {
    eddl::layer myLayer;
    if( type == "Input") {
        myLayer = static_cast<LInput *>(l);
    } else if (type == "Dense") {
        myLayer = static_cast<LDense *>(l);
    } else if (
        type == "Activation" ||
        type == "Softmax" ||
        type == "ReLu" ||
        type == "LeakyReLu" ||
        type == "Sigmoid"
    ) {
        myLayer = static_cast<LActivation *>(l);
    } else if (type == "Convolution") {
        myLayer = static_cast<LConv *>(l);
    } else if (
        type == "MaxPool" ||
        type == "GlobalMaxPool"
    ) {
        myLayer = static_cast<LMaxPool *>(l);
    } else if (
        type == "Reshape" ||
        type == "Flatten"
    ) {
        myLayer = static_cast<LReshape *>(l);
    } else if (type == "Concat") {
        myLayer = static_cast<LConcat *>(l);
    } else if (type == "BatchNormalization") {
        myLayer = static_cast<LBatchNorm *>(l);
    } else if (type == "UpSampling") {
        myLayer = static_cast<LUpSampling *>(l);
    }
    return myLayer;
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

template <class T, class T_ptr, class Func>
void fillVectorWithTypes(std::vector<T> &vector, const T_ptr* arr, int arr_count, Func func, std::vector<string> types) {
    std::vector<T_ptr> in_vector(arr, arr + arr_count);
    for(int i = 0; i < in_vector.size(); i++) {
        vector.push_back(func(in_vector[i], types[i]));
    }
}


extern "C" {

    // ---- TENSOR ----
    CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor(const int* shape, int shape_count, float *data) {
        const std::vector<int> shape_vector(shape, shape + shape_count);
        // Either 'DEV_CPU' or `DEV_GPU`
        // DEV_CPU - 0,DEV_GPU = 1000
        return new Tensor(shape_vector, data, 0);
    }

    CEDDLL_API tensor_ptr CALLING_CONV ceddl_tensor_load(const char* fname) {
        const std::string filename = string(fname);
        return Tensor::load(filename);
    }

    CEDDLL_API float* CALLING_CONV ceddl_tensor_getptr(tensor_ptr t) {
        tensor t1 = transformTensor(t);
        return t1->ptr;
    }

    // ---- TENSOR OPERATIONS ----
    CEDDLL_API void CALLING_CONV ceddl_div(tensor_ptr t, float v) {
        tensor t1 = transformTensor(t);
        return t1->div_(v);
    }

    CEDDLL_API int CALLING_CONV ceddl_ndim(tensor_ptr t) {
        tensor t1 = transformTensor(t);
        return t1->ndim;
    }

    CEDDLL_API int CALLING_CONV ceddl_size(tensor_ptr t) {
        tensor t1 = transformTensor(t);
        return t1->size;
    }

    CEDDLL_API void CALLING_CONV ceddl_print(tensor_ptr t) {
        tensor t1 = transformTensor(t);
        t1->print();
    }
    
    CEDDLL_API void CALLING_CONV ceddl_info(tensor_ptr t) {
        tensor t1 = transformTensor(t);
        t1->info();
    }

    CEDDLL_API tensor_ptr CALLING_CONV ceddl_select(tensor_ptr t, const char** indices, int indices_count) {
        std::vector<string> indices_vector = std::vector<string>();
        fillVector(indices_vector, indices, indices_count, transformString);
        tensor t1 = transformTensor(t);
        return t1->select(indices_vector);
    }

    ///////////////////////////////////////
    //  MODEL METHODS
    ///////////////////////////////////////
    
    // Load onnx format data
    CEDDLL_API model_ptr CALLING_CONV ceddl_import_onnx(const char* path, const int* input_shape, int input_shape_count) {
        std::string path_string = string(path);
        const std::vector<int> shape_vector(input_shape, input_shape + input_shape_count);
        return import_net_from_onnx_file(path_string, shape_vector);
    }
    
    // Creation
    CEDDLL_API model_ptr CALLING_CONV ceddl_Model(
        layer_ptr* in, int in_count, const char** in_types,
        layer_ptr* out, int out_count, const char** out_types
    ) {
        std::vector<string> in_types_vector = std::vector<string>();
        std::vector<string> out_types_vector = std::vector<string>();
        fillVector(in_types_vector, in_types, in_count, transformString);
        fillVector(out_types_vector, out_types, out_count, transformString);

        std::vector<eddl::layer> in_vector = std::vector<eddl::layer>();
        std::vector<eddl::layer> out_vector = std::vector<eddl::layer>();
        fillVectorWithTypes(in_vector, in, in_count, transformLayer, in_types_vector);
        fillVectorWithTypes(out_vector, out, out_count, transformLayer, out_types_vector);
        return eddl::Model(in_vector, out_vector);
    }

    CEDDLL_API void CALLING_CONV ceddl_build(
        model_ptr net,
        optimizer_ptr o, const char* type,
        const char** lo, int lo_count,
        const char** me, int me_count,
        compserv_ptr cs
    ) {
        const std::string type_str = string(type);
        std::vector<string> lo_vector = std::vector<string>();
        std::vector<string> me_vector = std::vector<string>();
        fillVector(lo_vector, lo, lo_count, transformString);
        fillVector(me_vector, me, me_count, transformString);
        Optimizer* myOptimizer;
        if (type_str == "SGD") {
            myOptimizer = static_cast<SGD *>(o);
        } else if (type_str == "Adam") {
            myOptimizer = static_cast<Adam *>(o);
        } else if (type_str == "AdaDelta") {
            myOptimizer = static_cast<AdaDelta *>(o);
        } else if (type_str == "Adagrad") {
            myOptimizer = static_cast<Adagrad *>(o);
        } else if (type_str == "Adamax") {
            myOptimizer = static_cast<Adamax *>(o);
        } else if (type_str == "Nadam") {
            myOptimizer = static_cast<Nadam *>(o);
        } else {
            // type_str == "RMSProp"
            myOptimizer = static_cast<RMSProp *>(o);
        }
        eddl::build(static_cast<eddl::model>(net), myOptimizer, lo_vector, me_vector, static_cast<eddl::compserv>(cs));
    }

    // Computing services
    CEDDLL_API compserv_ptr CALLING_CONV ceddl_CS_CPU(int th) {
        return eddl::CS_CPU(th);
    }

    // Info and logs
    CEDDLL_API void CALLING_CONV ceddl_setlogfile(model_ptr m, const char* fname) {
        const std::string fname_str = string(fname);
        eddl::setlogfile(static_cast<eddl::model>(m), fname_str);
    }

    CEDDLL_API void CALLING_CONV ceddl_summary(model_ptr m) {
        eddl::summary(static_cast<eddl::model>(m));
    }

    CEDDLL_API void CALLING_CONV ceddl_plot(model_ptr m, const char* fname) {
        const std::string fname_str = string(fname);
        eddl::plot(static_cast<eddl::model>(m), fname_str);
    }

    // Serialization
    CEDDLL_API void CALLING_CONV ceddl_load(model_ptr m, const char* fname) {
        const std::string fname_str = string(fname);
        eddl::load(static_cast<eddl::model>(m), fname_str);
    }

    CEDDLL_API void CALLING_CONV ceddl_save(model_ptr m, const char* fname) {
        const std::string fname_str = string(fname);
        eddl::save(static_cast<eddl::model>(m), fname_str);
    }

    // Optimizer
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
    
    // Training and Evaluation
    CEDDLL_API void CALLING_CONV ceddl_fit(model_ptr m,
        const tensor_ptr* in, int in_count,
        const tensor_ptr* out, int out_count,
        int batch, int epochs
    ) {
        std::vector<tensor> in_vector = std::vector<tensor>();
        std::vector<tensor> out_vector = std::vector<tensor>();
        fillVector(in_vector, in, in_count, transformTensor);
        fillVector(out_vector, out, out_count, transformTensor);
        eddl::fit(static_cast<eddl::model>(m), in_vector, out_vector, batch, epochs);
    }

    CEDDLL_API void CALLING_CONV ceddl_evaluate(model_ptr m,
        const tensor_ptr* in, int in_count,
        const tensor_ptr* out, int out_count
    ) {
        std::vector<tensor> in_vector = std::vector<tensor>();
        std::vector<tensor> out_vector = std::vector<tensor>();
        fillVector(in_vector, in, in_count, transformTensor);
        fillVector(out_vector, out, out_count, transformTensor);
        eddl::evaluate(static_cast<eddl::model>(m), in_vector, out_vector);
    }

    CEDDLL_API void CALLING_CONV ceddl_forward(model_ptr m, const tensor_ptr* in, int in_count) {
        std::vector<tensor> in_vector = std::vector<tensor>();
        fillVector(in_vector, in, in_count, transformTensor);
        eddl::forward(static_cast<eddl::model>(m), in_vector);
    }

    // loss and metrics methods
    CEDDLL_API metric_ptr CALLING_CONV ceddl_getMetric(const char* type) {
        const std::string type_str = string(type);
        return eddl::getMetric(type_str);
    }

    CEDDLL_API float CALLING_CONV ceddl_getMetricValue(metric_ptr metric, const char* type, tensor_ptr tensorT, tensor_ptr tensorY) {
        const std::string type_str = string(type);
        Metric* myMetric;
        if (type_str == "mse" || type_str == "mean_squared_error") {
            myMetric = static_cast<MMeanSquaredError *>(metric);
        } else if (type_str == "categorical_accuracy" || type_str == "accuracy") {
            myMetric = static_cast<MCategoricalAccuracy *>(metric);
        } else if (type_str == "mean_absolute_error") {
            myMetric = static_cast<MMeanAbsoluteError *>(metric);
        } else {
            // type_str == "mean_relative_error"
            myMetric = static_cast<MMeanRelativeError *>(metric);
        }
        return myMetric->value(transformTensor(tensorT), transformTensor(tensorY));
    }
    
    ///////////////////////////////////////
    //  LAYERS
    ///////////////////////////////////////

    // Get first output layer
    CEDDLL_API layer_ptr CALLING_CONV ceddl_GetOut(model_ptr net){
        layer_ptr outlayer = eddl::getOut(static_cast<eddl::model>(net))[0];
        return outlayer;
    }

    // Core Layers
    CEDDLL_API layer_ptr CALLING_CONV ceddl_Activation(
        layer_ptr parent, const char* parent_type,
        char* activation,
        float* params, int params_size,
        char* name
    ) {
        const std::string activation_str = string(activation);
        const std::vector<float> param_vector(params, params + params_size);
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::Activation(transformLayer(parent, parent_type_str), activation_str, param_vector, name_str);
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Sigmoid(layer_ptr parent, const char* parent_type, char* name) {
        const std::string parent_type_str = string(parent_type);
        return eddl::Sigmoid(transformLayer(parent, parent_type_str), name);
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Softmax(layer_ptr parent, const char* parent_type, char* name) {
        const std::string parent_type_str = string(parent_type);
        return eddl::Softmax(transformLayer(parent, parent_type_str), -1, name);
    }
    
    CEDDLL_API layer_ptr CALLING_CONV ceddl_ReLu(layer_ptr parent, const char* parent_type) {
        const std::string parent_type_str = string(parent_type);
        return eddl::ReLu(transformLayer(parent, parent_type_str));
    }
    
    CEDDLL_API layer_ptr CALLING_CONV ceddl_LeakyReLu(layer_ptr parent, const char* parent_type) {
        const std::string parent_type_str = string(parent_type);
        return eddl::LeakyReLu(transformLayer(parent, parent_type_str));
    }
    
    CEDDLL_API layer_ptr CALLING_CONV ceddl_Conv(
        layer_ptr parent, const char* parent_type,
        int filters,
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
        const std::string parent_type_str = string(parent_type);
        return eddl::Conv(
            transformLayer(parent, parent_type_str), filters,
            kernel_size_vector, strides_vector, padding_str, use_bias, groups,
            dilation_rate_vector, name_str
        );
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Dense(
        layer_ptr parent, const char* parent_type,
        int num_dim,
        bool use_bias,
        const char* name
    ) {
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::Dense(transformLayer(parent, parent_type_str), num_dim, use_bias, name_str);
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Input(const int* shape, int shape_count, const char* name) {
        const std::vector<int> shape_vector(shape, shape + shape_count);
        const std::string name_str = string(name);
        return eddl::Input(shape_vector, name_str);
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_UpSampling(
        layer_ptr parent, const char* parent_type,
        const int* size, int size_count,
        const char* interpolation,
        const char* name
    ) {
        const std::vector<int> size_vector(size, size + size_count);
        const std::string interpolation_str = string(interpolation);
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::UpSampling(transformLayer(parent, parent_type_str), size_vector, interpolation_str, name_str);	
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Reshape(
        layer_ptr parent, const char* parent_type,
        const int* shape, int shape_count,
        const char* name
    ) {
        const std::vector<int> shape_vector(shape, shape + shape_count);
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::Reshape(transformLayer(parent, parent_type_str), shape_vector, name_str);
    }

    CEDDLL_API layer_ptr CALLING_CONV ceddl_Flatten(layer_ptr parent, const char* parent_type, const char* name) {
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::Flatten(transformLayer(parent, parent_type_str), name_str);
    }

    // ---- MERGE LAYERS ----

    CEDDLL_API model_ptr CALLING_CONV ceddl_Concat(
        layer_ptr* in, int in_count, const char** in_types
    ) {
        std::vector<string> in_types_vector = std::vector<string>();
        fillVector(in_types_vector, in_types, in_count, transformString);
        std::vector<eddl::layer> in_vector = std::vector<eddl::layer>();
        fillVectorWithTypes(in_vector, in, in_count, transformLayer, in_types_vector);
        return eddl::Concat(in_vector);
    }

    // ---- NOISE LAYERS ----

    // ---- NORMALIZATION LAYERS ----
    
    CEDDLL_API tensor_ptr CALLING_CONV ceddl_BatchNormalization(layer_ptr layer, const char* layer_type) {
        return eddl::BatchNormalization(transformLayer(layer, layer_type));
    }

    // ---- OPERATOR LAYERS ----

    // ---- REDUCTION LAYERS ----

    // ---- GENERATOR LAYERS ----

    // ---- POOLING LAYERS ----

    CEDDLL_API layer_ptr CALLING_CONV ceddl_GlobalMaxPool(layer_ptr parent, const char* parent_type, const char* name) {
        const std::string name_str = string(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::GlobalMaxPool(transformLayer(parent, parent_type_str), name_str);
    }
    
    CEDDLL_API layer_ptr CALLING_CONV ceddl_MaxPool(
        layer_ptr parent, const char* parent_type,
        const int* pool_size, int pool_size_count,
        const int* strides, int strides_count,
        const char* padding, const char* name
    ) {
        const std::vector<int> pool_size_vector(pool_size, pool_size + pool_size_count);
        const std::vector<int> strides_vector(strides, strides + strides_count);
        const std::string padding_str(padding);
        const std::string name_str(name);
        const std::string parent_type_str = string(parent_type);
        return eddl::MaxPool(transformLayer(parent, parent_type_str), pool_size_vector, strides_vector, padding_str, name_str);
    }

    // Recurrent Layers

    //////////////////////////////
    // Layers Methods
    //////////////////////////////

    ////////////////////////////////////
    // Manage Tensors inside Layers
    ////////////////////////////////////

    CEDDLL_API tensor_ptr CALLING_CONV ceddl_GetOutput(layer_ptr layer, const char* layer_type) {
        return eddl::getOutput(transformLayer(layer, layer_type));
    }

    ///////////////////////////////////////
    //  INITIALIZERS
    ///////////////////////////////////////

    ///////////////////////////////////////
    //  REGULARIZERS
    ///////////////////////////////////////

    ///////////////////////////////////////
    //  DATASETS
    ///////////////////////////////////////
    CEDDLL_API void CALLING_CONV ceddl_download_mnist() {
        eddl::download_mnist();
    }
    
    CEDDLL_API void CALLING_CONV ceddl_download_cifar10() {
        eddl::download_cifar10();
    }
}
    