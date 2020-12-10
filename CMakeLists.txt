cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(mmvae
  VERSION 0.1
  DESCRIPTION "")

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

###########################
# Typical Gzipp'ed stream #
###########################

add_library(gzslib STATIC
  src/utils/gzstream.cc
  include/utils/gzstream.hh
  )

target_include_directories(gzslib PUBLIC
  include/utils
  )

####################
# BGZipp'ed stream #
####################

add_library(bgzf STATIC
  src/ext/tabix/bgzf.c
  src/ext/tabix/kstring.c
  include/ext/tabix/bgzf.h
  include/ext/tabix/kstring.h
  )

target_include_directories(bgzf PUBLIC
  include/ext/tabix
  )

add_library(bgzslib STATIC
  src/utils/bgzstream.cc
  include/utils/bgzstream.hh
  )

target_include_directories(bgzslib PUBLIC
  include/utils
  include/ext/tabix
  )

##############
# Executable #
##############

file(GLOB main_files "src/*_main.cc")

foreach(file ${main_files})
  get_filename_component(fname ${file} NAME "src/")
  string(REPLACE "_main.cc" "" name ${fname})
  string(REPLACE "_vae" "" model ${name})
  add_executable(${name}
    ${file}
    include/mmvae.hh
    include/mmvae_io.hh
    include/mmvae_alg.hh
    include/models/${model}.hh
    include/modules/angular.hh)
  target_include_directories(${name} PUBLIC include include/models include/modules)
  target_link_libraries(${name} "${TORCH_LIBRARIES}")
  target_link_libraries(${name} ZLIB::ZLIB gzslib bgzslib bgzf)
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 14)
endforeach(file ${main_files})
