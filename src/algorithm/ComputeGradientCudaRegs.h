//
// Created by gonciarz on 3/23/18.
//

#ifndef LIBAPR_COMPUTEGRADIENTCUDAREGS_H
#define LIBAPR_COMPUTEGRADIENTCUDAREGS_H

#include "data_structures/Mesh/MeshData.hpp"

void cudaFilterBsplineYdirectionRegs(MeshData<float> &input);

#endif //LIBAPR_COMPUTEGRADIENTCUDAREGS_H
