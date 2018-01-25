//
//  Bevan Cheeseman 2018
//
//  The Pulling Scheme using the full Particle Cell Pyramid as a tree structure.
//
//
//  This code was originally optimized for the OpenMP version by Matteusz Susik in (2016)
//
//

#ifndef PARTPLAY_PULLING_SCHEME_HPP
#define PARTPLAY_PULLING_SCHEME_HPP

#include <cassert>
#include <src/data_structures/APR/APRIterator.hpp>

#include "src/data_structures/Mesh/MeshData.hpp"
#include "src/data_structures/APR/APR.hpp"

#ifdef HAVE_OPENMP
	#include "omp.h"
#endif

#define EMPTY 0
#define SEED_TYPE 1
#define BOUNDARY_TYPE 2
#define FILLER_TYPE 3
#define ASCENDANT 8
#define SEEDASCENDANT 9
#define PROPOGATE 15
#define ASCENDANTNEIGHBOUR 16

#define MOORE 1
#define VONNEUMANN 0

#define NEIGHBOURLOOP(jn,in,kn, boundaries) \
for(jn = boundaries[0][0]; jn < boundaries[0][1]; jn++) \
    for(in = boundaries[1][0]; in < boundaries[1][1]; in++) \
        for(kn = boundaries[2][0]; kn < boundaries[2][1]; kn++)


#define CHILDRENLOOP(jn,in,kn, children_boundaries) \
for(jn = j * 2; jn < j * 2 + children_boundaries[0]; jn++) \
    for(in = i * 2; in < i * 2 + children_boundaries[1]; in++) \
        for(kn = k * 2; kn < k * 2 + children_boundaries[2]; kn++)

// don't try to optimize check boundaries - every check is needed due to parallelism

#define CHECKBOUNDARIES(axis,var,limit,boundaries) \
    if (var == 0) { \
        boundaries[axis][0] = 0;\
    } else {\
        boundaries[axis][0] = -1;\
    }\
    if (var == limit) {\
        boundaries[axis][1] = 1;\
    } else {\
        boundaries[axis][1] = 2;\
    }


class PullingScheme {

/*
 *  Declerations
 */

public:

    std::vector<MeshData<uint8_t>> particle_cell_tree;
    unsigned int l_min;
    unsigned int l_max;

    template<typename T>
    void fill(float k, MeshData<T> &input);

    void pulling_scheme_main();

    template<typename T>
    void initialize_particle_cell_tree(APR<T>& apr);

private:
    void check_boundaries(short axis, int var, int limit, short (&boundaries)[3][2]);

    void set_ascendant_neighbours(int level);

    void set_filler(int level);

    void fill_neighbours(int level);

    void fill_parent(int j, int i, int k, int x_num, int y_num, int new_level);

};
/*
 *  Definitions
 */

template<typename T>
void PullingScheme::initialize_particle_cell_tree(APR<T>& apr)
{   //
    //  Initializes the particle cell tree structure
    //
    //  Contains pc up to l_max - 1,
    //

    this->l_max = apr.level_max() - 1;
    this->l_min = apr.level_min();
    //make so you can reference the array as l
    particle_cell_tree.resize(l_max + 1);

    for(int l = l_min; l < (l_max + 1) ;l ++){
        particle_cell_tree[l].initialize(ceil((1.0*apr.apr_access.org_dims[0])/pow(2.0,1.0*l_max - l + 1)),
                                         ceil((1.0*apr.apr_access.org_dims[1])/pow(2.0,1.0*l_max - l + 1)),
                                         ceil((1.0*apr.apr_access.org_dims[2])/pow(2.0,1.0*l_max - l + 1)), EMPTY);
    }
}


void PullingScheme::pulling_scheme_main()
{
    //
    //  Bevan Cheeseman 2016
    //
    //  The Pulling Scheme for forming the Optimal Valid Particle Cell set from the Local Particle Cell set L
    //
    //  Implimented as discussed in Cheeseman et al. 2017 for full description.
    //
    //  Generates the implied resolution function that is used to sample the image in the APR.
    //


    //loop over all levels from l_max to l_min
    for (int level = l_max; l_min <= level; level--) {

        if (level != l_max) {

            set_ascendant_neighbours(level); //step 1 and step 2.

            set_filler(level); // step 3.

        }
        fill_neighbours(level); // step 4.

    }

}

template<typename T>
void PullingScheme::fill(const float k, MeshData<T>& input)
{
    //
    //  Bevan Cheeseman 2016
    //
    //  Updates the hash table from the down sampled images
    //

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    int temp;
    int i,q;

    auto &topvec = particle_cell_tree[l_max].mesh;

    if (k == l_max){
        // k_max loop, has to include
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,q,temp)
#endif
        for(int j = 0;j < z_num;j++){
            for(i = 0;i < x_num;i++){
                for (q = 0; q < (y_num);q++){

                    temp = input.mesh[j*x_num*y_num + i*y_num + q] >= k;

                    if ( temp ) {
                        topvec[j*x_num*y_num + i*y_num + q] = SEED_TYPE;
                    }
                }
            }
        }



    } else if (k == l_min){
        // k_max loop, has to include
#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,q,temp) if(z_num*x_num*y_num > 100000)
#endif
        for(int j = 0;j < z_num;j++){
            for(i = 0;i < x_num;i++){
                for (q = 0; q < (y_num);q++){

                    temp = input.mesh[j*x_num*y_num + i*y_num + q] <= k;

                    if ( temp ) {
                        particle_cell_tree[l_min].mesh[j*x_num*y_num + i*y_num + q] = SEED_TYPE;
                    }
                }
            }
        }



    } else{
        // other k's

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,q,temp) if(z_num*x_num*y_num > 100000)
#endif
        for(int j = 0;j < z_num;j++){
            for(i = 0;i < x_num;i++){
#ifndef _MSC_VER
#ifdef HAVE_OPENMP
	#pragma omp simd
#endif
#endif
                for (q = 0; q < y_num;q++){

                    temp = input.mesh[j*x_num*y_num + i*y_num + q] == k;

                    if (temp) {
                        particle_cell_tree[k].mesh[j*x_num*y_num + i*y_num + q] = SEED_TYPE;
                    }
                }
            }
        }
    }
}



void PullingScheme::check_boundaries(short axis, int var, int limit, short (&boundaries)[3][2])
{

    if (var == 0) {
        boundaries[axis][0] = 0;
        boundaries[axis][1] = 2;
    } else if (var == 1) {
        boundaries[axis][0] = -1;
    }
    if (var == limit) {
        boundaries[axis][1] = 1;
    }
}


void PullingScheme::set_ascendant_neighbours(int level)
{

    const int x_num = particle_cell_tree[level].x_num;
    const int y_num = particle_cell_tree[level].y_num;
    const int z_num = particle_cell_tree[level].z_num;

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};

    int i,k,jn,in,kn,neighbour_index,index;

    uint8_t status;

    // loop unrolling in order to avoid concurrent write
    for(int out = 0; out < std::min(3,z_num); out ++) {

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        if(z_num * x_num * y_num > 100000) schedule(static)
#endif
        for (int j = out; j < z_num; j += 3) {

            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

            for (i = 0; i < x_num; i++) {
                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                index = j * x_num * y_num + i * y_num;

                for (k = 0; k < y_num; k++) {

                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    status = particle_cell_tree[level].mesh[index + k];

                    if (status == ASCENDANT) {
                        NEIGHBOURLOOP(jn, in, kn, boundaries) {

                                    neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                    if (particle_cell_tree[level].mesh[neighbour_index] == EMPTY) {
                                        // type is EMPTY
                                        particle_cell_tree[level].mesh[neighbour_index] = ASCENDANTNEIGHBOUR;
                                    }

                                    if (particle_cell_tree[level].mesh[neighbour_index] == SEED_TYPE) {
                                        // type is SEED
                                        particle_cell_tree[level].mesh[neighbour_index] = PROPOGATE;
                                    }
                                }
                    }
                }
            }
        }
    }
}

void PullingScheme::set_filler(int level)
{
    short children_boundaries[3] = {2,2,2};
    const int x_num = particle_cell_tree[level].x_num;
    const int y_num = particle_cell_tree[level].y_num;
    const int z_num = particle_cell_tree[level].z_num;

    int prev_x_num = particle_cell_tree[level + 1].x_num;
    int prev_y_num = particle_cell_tree[level + 1].y_num;
    int prev_z_num = particle_cell_tree[level + 1].z_num;

    int i, k, jn, in, kn, children_index, index, parts=0;
    uint8_t children_status, status;

#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) \
        private(i,k,children_index,jn,in,kn,children_status,status,index) \
        if(z_num * x_num * y_num > 10000) firstprivate(level, children_boundaries)
#endif
    for(int j = 0; j < z_num; j++) {

        if( j == z_num - 1 && prev_z_num % 2 ) {
            children_boundaries[0] = 1;
        }

        for ( i = 0; i < x_num; i++) {

            if( i == x_num - 1 && prev_x_num % 2 ) {
                children_boundaries[1] = 1;
            } else if( i == 0 ){
                children_boundaries[1] = 2;
            }

            index = j * x_num * y_num + i * y_num;


            for ( k = 0; k < y_num; k++) {

                if( k == y_num - 1 && prev_y_num % 2 ) {
                    children_boundaries[2] = 1;
                } else if( k == 0 ){
                    children_boundaries[2] = 2;
                }

                status = particle_cell_tree[level].mesh[index + k];

                if(status == ASCENDANTNEIGHBOUR || status == PROPOGATE) {

                    // go down, and set empty children to FILLER
                    CHILDRENLOOP(jn, in, kn, children_boundaries) {

                                children_index = jn * prev_x_num * prev_y_num + in * prev_y_num + kn;
                                children_status = particle_cell_tree[level + 1].mesh[children_index];
                                if(children_status == EMPTY) {
                                    particle_cell_tree[level + 1].mesh[children_index] = FILLER_TYPE;
                                }
                            }

                }

            }
        }
    }

}

void PullingScheme::fill_neighbours(int level)
{
    const int x_num = particle_cell_tree[level].x_num;
    const int y_num = particle_cell_tree[level].y_num;
    const int z_num = particle_cell_tree[level].z_num;

    int j,i,k,neighbour_index,jn,in,kn,index,parts = 0;
    uint8_t status;

    short boundaries[3][2] = {{0,2},{0,2},{0,2}};

    // loop unrolling in order to avoid concurrent write
    for(int out = 0; out < std::min(3,z_num); out ++) {


#ifdef HAVE_OPENMP
	#pragma omp parallel for default(shared) private(j,i,k,neighbour_index,jn,in,kn,status,index) firstprivate(boundaries) \
        if(z_num * x_num * y_num > 100000)
#endif
        for (int j = out; j < z_num; j += 3) {

            CHECKBOUNDARIES(0, j, z_num - 1, boundaries);

            for (i = 0; i < x_num; i++) {

                CHECKBOUNDARIES(1, i, x_num - 1, boundaries);
                index = j * x_num * y_num + i * y_num;

                for (k = 0; k < y_num; k++) {

                    CHECKBOUNDARIES(2, k, y_num - 1, boundaries);
                    status = particle_cell_tree[level].mesh[index + k];

                    if (status == SEED_TYPE || status == PROPOGATE) {

                        NEIGHBOURLOOP(jn, in, kn, boundaries) {
                                    neighbour_index = index + jn * x_num * y_num + in * y_num + kn + k;

                                    if (particle_cell_tree[level].mesh[neighbour_index] == EMPTY) {
                                        particle_cell_tree[level].mesh[neighbour_index] = BOUNDARY_TYPE;
                                    }
                                }
                        parts += 8;

                        fill_parent(j, i, k, x_num, y_num, level - 1);
                    } else if (status == ASCENDANT) {
                        fill_parent(j, i, k, x_num, y_num, level - 1);
                    }
                }
            }
        }
    }

}

void PullingScheme::fill_parent(int j, int i, int k, int x_num, int y_num, int new_level)
{

    if(new_level >= l_min) {
        int new_x_num = ((x_num + 1) / 2);
        int new_y_num = ((y_num + 1) / 2);
        int new_index = (j / 2) * new_x_num * new_y_num + (i / 2) * new_y_num + (k / 2);

        if (particle_cell_tree[new_level].mesh[new_index] != SEED_TYPE) {
            particle_cell_tree[new_level].mesh[new_index] = ASCENDANT;
        }
    }
}







#endif //PARTPLAY_PULLING_SCHEME_HPP
