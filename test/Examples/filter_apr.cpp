#include <algorithm>
#include <iostream>

#include "filter_apr.h"
#include "../../src/data_structures/meshclass.h"
#include "../../src/io/readimage.h"

#include "../../src/algorithm/gradient.hpp"
#include "../../src/data_structures/particle_map.hpp"
#include "../../src/data_structures/Tree/PartCellBase.hpp"
#include "../../src/data_structures/Tree/PartCellStructure.hpp"
#include "../../src/data_structures/Tree/ParticleDataNew.hpp"
#include "../../src/algorithm/level.hpp"
#include "../../src/io/writeimage.h"
#include "../../src/io/write_parts.h"
#include "../../src/io/partcell_io.h"

#include "../../test/utils.h"

#include "../../src/numerics/misc_numerics.hpp"
#include "../../src/numerics/filter_numerics.hpp"




bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv, Part_rep& part_rep){
    
    cmdLineOptions result;
    
    if(argc == 1) {
        std::cerr << "Usage: \"pipeline -i inputfile -d directory [-t] [-o outputfile]\"" << std::endl;
        exit(1);
    }
    
    if(command_option_exists(argv, argv + argc, "-i"))
    {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    
    if(command_option_exists(argv, argv + argc, "-d"))
    {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    
    if(command_option_exists(argv, argv + argc, "-o"))
    {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    if(command_option_exists(argv, argv + argc, "-gt"))
    {
        result.gt = std::string(get_command_option(argv, argv + argc, "-gt"));
    }
    
    if(command_option_exists(argv, argv + argc, "-t"))
    {
        part_rep.timer.verbose_flag = true;
    }
    
    return result;
    
}

int main(int argc, char **argv) {
    
    Part_rep part_rep;
    
    // INPUT PARSING
    
    cmdLineOptions options = read_command_line_options(argc, argv, part_rep);
    
    // APR data structure
    PartCellStructure<float,uint64_t> pc_struct;
    
    // Filename
    std::string file_name = options.directory + options.input;
    
    // Read the apr file into the part cell structure
    read_apr_pc_struct(pc_struct,file_name);
    
    
    //////////////////////////////////
    //
    //  Different access and filter test examples
    //
    //////////////////////////////////
    
    //set up some new structures used in this test
    AnalysisData analysis_data;

    float num_repeats = 1;

    //Get neighbours (linear)

    //particles
    //
   // particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);

    particle_linear_neigh_access(pc_struct,num_repeats,analysis_data);



    lin_access_parts(pc_struct);

    ParticleDataNew<float, uint64_t> part_new;

    part_new.initialize_from_structure(pc_struct);

    PartCellData<uint64_t> pc_data;
    part_new.create_pc_data_new(pc_data);

    uint64_t counter = 0;


    for(uint64_t depth = (pc_data.depth_min);depth <= pc_data.depth_max;depth++) {
        //loop over the resolutions of the structure
        for(int i = 0;i < pc_data.data[depth].size();i++){

            counter += pc_data.data[depth][i].size();
        }

    }

    std::cout << counter << std::endl;
    std::cout << pc_struct.get_number_parts() << std::endl;

    //particle_linear_neigh_access_alt_1(pc_struct);

    //pixels
   // pixels_linear_neigh_access(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],num_repeats,analysis_data);


    //Get neighbours (random access)

    //particle_random_access(pc_struct,analysis_data);

   // pixel_neigh_random(pc_struct,pc_struct.org_dims[0],pc_struct.org_dims[1],pc_struct.org_dims[2],analysis_data);


    // Filtering

    uint64_t filter_offset = 10;

    //apr_filter_full(pc_struct,filter_offset,num_repeats,analysis_data);






    //new_filter_part(pc_struct,filter_offset,num_repeats,analysis_data);

    //interp_slice<float,float>(pc_struct,pc_struct.part_data.particle_data,dir,num);

    //get_slices<float>(pc_struct);

//    Mesh_data<uint16_t> output;
//
//    Part_timer timer;
//
//    timer.verbose_flag = true;
//
//
//    std::vector<float> filter;
//
//    filter = create_dog_filter<float>(filter_offset,1.5,3);
//
//    //filter = {-1,0,1};
//
//    ExtraPartCellData<float> filter_output;
//
//    filter_output = filter_apr_by_slice<float>(pc_struct,filter,analysis_data,num_repeats,true);
//
//    Mesh_data<float> input_image;
//
//    pc_struct.interp_parts_to_pc(input_image,pc_struct.part_data.particle_data);
//
//    Mesh_data<float> output_image;
//
//    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);
//
//    for (int k = 0; k < output_image.mesh.size(); ++k) {
//        output_image.mesh[k] = 10 * fabs(output_image.mesh[k]);
//    }
//    debug_write(output_image,"img_filter_full");
//
//    Mesh_data<uint16_t> input_image_;
//
//    load_image_tiff(input_image_,options.gt);
//
//    input_image = input_image_.to_type<float>();
//
//    output_image =  pixel_filter_full(input_image,filter,num_repeats,analysis_data);
//
//    for (int k = 0; k < output_image.mesh.size(); ++k) {
//        output_image.mesh[k] = 10 * fabs(output_image.mesh[k]);
//    }
//    debug_write(output_image,"img_filter_org");
//
//    ExtraPartCellData<float> filter_output_mesh;
//
//    filter_output_mesh = filter_apr_input_img<float>(input_image,pc_struct,filter,analysis_data,num_repeats,true);

}


