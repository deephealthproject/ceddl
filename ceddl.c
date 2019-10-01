/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Ynse Hoornenborg: ynse.hoornenborg@philips.com
//      Roberto Paredes: rparedes@prhlt.upv.es
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

}